import os
import tempfile
import time

import pandas as pd
from fastapi import APIRouter, UploadFile, File, Form, HTTPException

from app.services.parser import load_resume_chunks
from app.services.matcher import (
    split_jd_into_lines,
    build_structured_jd_requirements,
    find_missing_items,
    calculate_match_score,
)
from app.services.rewriter import generate_rewrite_suggestions

router = APIRouter()


def normalize_missing_items_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Make sure the dataframe has the columns expected by downstream functions.
    This prevents KeyError issues like: 'resume_strength'
    """
    if df is None:
        df = pd.DataFrame()

    df = df.copy()

    rename_map = {
        "strength": "resume_strength",
        "match_strength": "resume_strength",
        "evidence": "resume_evidence",
        "best_evidence": "resume_evidence",
        "matched_status": "matched",
        "is_matched": "matched",
        "requirement": "jd_requirement",
        "jd_line": "jd_requirement",
    }

    for old_col, new_col in rename_map.items():
        if old_col in df.columns and new_col not in df.columns:
            df.rename(columns={old_col: new_col}, inplace=True)

    expected_defaults = {
        "jd_requirement": "",
        "category": "general",
        "matched": False,
        "resume_evidence": "",
        "resume_strength": "missing",
        "missing_reason": "",
        "gap_type": "missing",
    }

    for col, default_value in expected_defaults.items():
        if col not in df.columns:
            df[col] = default_value

    df["matched"] = df["matched"].fillna(False)

    def normalize_strength(value):
        if pd.isna(value):
            return "missing"
        value = str(value).strip().lower()
        if value in ["strong", "medium", "weak", "missing"]:
            return value
        if value in ["high", "good"]:
            return "strong"
        if value in ["partial", "moderate", "some"]:
            return "medium"
        if value in ["low", "poor"]:
            return "weak"
        return "missing"

    df["resume_strength"] = df["resume_strength"].apply(normalize_strength)

    return df


def safe_calculate_match_score(df: pd.DataFrame) -> float:
    """
    Try original score function first.
    If it fails, compute a fallback score from matched rows.
    """
    try:
        return calculate_match_score(df)
    except Exception:
        if df is None or len(df) == 0:
            return 0.0

        if "matched" in df.columns:
            matched_count = int(df["matched"].fillna(False).astype(bool).sum())
            total_count = len(df)
            return round((matched_count / total_count) * 100, 2) if total_count > 0 else 0.0

        return 0.0


def safe_generate_rewrite_suggestions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Try original rewrite function first.
    If it fails, return a basic fallback suggestion dataframe.
    """
    try:
        result = generate_rewrite_suggestions(df)
        if result is None:
            return pd.DataFrame()
        return result
    except Exception:
        fallback_rows = []

        for _, row in df.iterrows():
            matched = bool(row.get("matched", False))
            if not matched:
                jd_requirement = str(row.get("jd_requirement", "")).strip()
                category = str(row.get("category", "general")).strip()
                reason = str(row.get("missing_reason", "")).strip()

                fallback_rows.append(
                    {
                        "jd_requirement": jd_requirement,
                        "category": category,
                        "suggestion": f"Add a resume bullet that demonstrates evidence for: {jd_requirement}",
                        "missing_reason": reason,
                    }
                )

        return pd.DataFrame(fallback_rows)


@router.post("/analyze")
async def analyze_resume(
    resume: UploadFile = File(...),
    job_description: str = Form(...)
):
    if not resume.filename or not resume.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF resumes are supported.")

    if not job_description or len(job_description.strip()) < 30:
        raise HTTPException(status_code=400, detail="Job description is too short.")

    temp_pdf_path = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            content = await resume.read()
            tmp.write(content)
            temp_pdf_path = tmp.name

        timings = {}

        start = time.time()
        resume_chunks = load_resume_chunks(temp_pdf_path)
        timings["resume_parse_seconds"] = round(time.time() - start, 2)

        start = time.time()
        jd_requirements = split_jd_into_lines(job_description)
        structured_jd_df = build_structured_jd_requirements(jd_requirements)
        timings["jd_structure_seconds"] = round(time.time() - start, 2)

        start = time.time()
        missing_items_df = find_missing_items(structured_jd_df, resume_chunks)
        missing_items_df = normalize_missing_items_df(missing_items_df)
        timings["matching_seconds"] = round(time.time() - start, 2)

        start = time.time()
        rewrite_df = safe_generate_rewrite_suggestions(missing_items_df)
        timings["rewrite_seconds"] = round(time.time() - start, 2)

        match_score = safe_calculate_match_score(missing_items_df)

        return {
            "success": True,
            "filename": resume.filename,
            "match_score": match_score,
            "structured_jd_count": int(len(structured_jd_df)),
            "missing_items_count": int(len(missing_items_df)),
            "rewrite_suggestions_count": int(len(rewrite_df)),
            "structured_jd": structured_jd_df.to_dict(orient="records"),
            "missing_items": missing_items_df.to_dict(orient="records"),
            "rewrite_suggestions": rewrite_df.to_dict(orient="records"),
            "timings": timings,
        }

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Backend error: {str(e)}")

    finally:
        if temp_pdf_path and os.path.exists(temp_pdf_path):
            os.remove(temp_pdf_path)
import re
import pandas as pd


CANONICAL_SKILL_MAP = {
    "machine learning": ["ml", "machine learning"],
    "statistics": ["statistics", "statistical analysis", "statistical"],
    "python": ["python"],
    "sql": ["sql"],
    "data analysis": ["data analysis", "data analytics", "analyze data", "analysis"],
    "experimentation": ["experiment", "experiments", "experimentation", "a/b testing", "ab testing"],
    "stakeholder communication": ["stakeholder", "cross-functional", "collaborate", "collaboration"],
    "algorithms": ["algorithm", "algorithms"],
    "data exploration": ["data exploration", "exploration", "eda", "exploratory data analysis"],
    "visualization": ["dashboard", "visualization", "visualisation", "chart", "graphs"],
}


def clean_text(text: str) -> str:
    text = str(text).strip()
    text = re.sub(r"\s+", " ", text)
    return text


def split_jd_into_lines(jd_text: str):
    """
    Convert job description into usable requirement lines.
    """
    raw_lines = jd_text.splitlines()
    lines = []

    for line in raw_lines:
        cleaned = line.strip(" -•\t\r")
        cleaned = clean_text(cleaned)

        if not cleaned:
            continue

        if len(cleaned) < 8:
            continue

        lower = cleaned.lower()

        skip_phrases = [
            "responsibilities",
            "qualifications",
            "microsoft is an equal opportunity employer",
            "this position will be open",
            "currently pursuing",
        ]

        if any(phrase in lower for phrase in skip_phrases):
            continue

        lines.append(cleaned)

    return lines


def extract_skills_from_requirement(requirement: str):
    """
    Very simple skill extraction based on canonical keyword map.
    """
    requirement_lower = requirement.lower()
    found_skills = []

    for canonical_skill, aliases in CANONICAL_SKILL_MAP.items():
        if any(alias in requirement_lower for alias in aliases):
            found_skills.append(canonical_skill)

    return found_skills


def build_structured_jd_requirements(jd_requirements):
    """
    Build a dataframe of JD requirement lines with extracted skills.
    """
    rows = []

    for i, req in enumerate(jd_requirements):
        rows.append({
            "jd_req_id": i,
            "jd_requirement": clean_text(req),
            "skills": extract_skills_from_requirement(req)
        })

    return pd.DataFrame(rows)


def match_requirement_against_resume(jd_requirement: str, skills: list, resume_text: str):
    """
    Rule-based matching for MVP.
    Later we can replace with embeddings/vector search.
    """
    req_lower = jd_requirement.lower()
    resume_lower = resume_text.lower()

    matched_skills = []
    missing_skills = []

    for skill in skills:
        aliases = CANONICAL_SKILL_MAP.get(skill, [skill])
        found = any(alias in resume_lower for alias in aliases)

        if found:
            matched_skills.append(skill)
        else:
            missing_skills.append(skill)

    # strength logic
    if skills:
        ratio = len(matched_skills) / len(skills)
    else:
        ratio = 0.0

    direct_overlap = any(token in resume_lower for token in req_lower.split() if len(token) > 5)

    if ratio >= 0.7 or (len(matched_skills) >= 2):
        strength = "strong"
        why = f"Matched skills: {', '.join(matched_skills)}" if matched_skills else "Good overlap found."
    elif ratio >= 0.3 or direct_overlap:
        strength = "weak"
        why = (
            f"Partial overlap found. Matched: {', '.join(matched_skills)}"
            if matched_skills else
            "Some textual overlap found, but evidence is limited."
        )
    else:
        strength = "missing"
        why = (
            f"Missing key skills: {', '.join(missing_skills)}"
            if missing_skills else
            "No strong evidence found in resume."
        )

    return {
        "resume_strength": strength,
        "matched_skills": matched_skills,
        "missing_skills": missing_skills,
        "why": why
    }


def find_missing_items(structured_jd_df, resume_chunks):
    """
    Compare each JD requirement against the resume text.
    """
    resume_text = "\n".join(chunk.page_content for chunk in resume_chunks)

    rows = []

    for _, row in structured_jd_df.iterrows():
        result = match_requirement_against_resume(
            jd_requirement=row["jd_requirement"],
            skills=row["skills"],
            resume_text=resume_text
        )

        rows.append({
            "jd_req_id": row["jd_req_id"],
            "jd_requirement": row["jd_requirement"],
            "skills": row["skills"],
            "resume_strength": result["resume_strength"],
            "matched_skills": result["matched_skills"],
            "missing_skills": result["missing_skills"],
            "why": result["why"]
        })

    return pd.DataFrame(rows)


def calculate_match_score(missing_items_df: pd.DataFrame) -> int:
    """
    Convert row-level strengths into a simple overall score.
    """
    if missing_items_df.empty:
        return 0

    score_map = {
        "strong": 1.0,
        "weak": 0.5,
        "missing": 0.0
    }

    values = [
        score_map.get(str(value).lower(), 0.0)
        for value in missing_items_df["resume_strength"].tolist()
    ]

    percentage = int((sum(values) / len(values)) * 100)
    return percentage
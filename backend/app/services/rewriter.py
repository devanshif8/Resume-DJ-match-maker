import os
import pandas as pd
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    api_key=OPENAI_API_KEY
)


def generate_rewrite_suggestions(missing_items_df: pd.DataFrame, top_n: int = 8):
    """
    Generate rewrite suggestions only for weak/missing rows.
    """
    rows = []

    rewrite_input_df = missing_items_df[
        missing_items_df["resume_strength"].astype(str).str.lower().isin(["missing", "weak", "low"])
    ].copy()

    rewrite_input_df = rewrite_input_df.head(top_n)

    if rewrite_input_df.empty:
        return pd.DataFrame(columns=[
            "jd_req_id",
            "jd_requirement",
            "resume_strength",
            "why",
            "rewritten_bullets"
        ])

    for _, row in rewrite_input_df.iterrows():
        jd_req = row["jd_requirement"]
        matched_skills = row.get("matched_skills", [])
        missing_skills = row.get("missing_skills", [])
        why = row.get("why", "")

        prompt = f"""
You are helping improve a student's resume for a data science internship.

Job requirement:
{jd_req}

Current analysis:
- Resume strength: {row["resume_strength"]}
- Matched skills: {matched_skills}
- Missing skills: {missing_skills}
- Why: {why}

Task:
Write 2 honest, ATS-friendly resume bullet suggestions that could better align the resume to this requirement.

Rules:
- Do not invent fake achievements.
- Keep bullets realistic for a student or internship candidate.
- Make them concise and professional.
- Return plain text bullets only.
"""

        try:
            response = llm.invoke(prompt)
            bullets = response.content.strip()
        except Exception as e:
            bullets = f"Rewrite failed: {str(e)}"

        rows.append({
            "jd_req_id": row["jd_req_id"],
            "jd_requirement": jd_req,
            "resume_strength": row["resume_strength"],
            "why": why,
            "rewritten_bullets": bullets
        })

    return pd.DataFrame(rows)
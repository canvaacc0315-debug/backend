# backend/ai_utils.py
import os
from enum import Enum
from typing import Optional

import requests
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_NAME = "models/gemini-2.5-pro"

GEMINI_URL = (
    f"https://generativelanguage.googleapis.com/v1beta/"
    f"{MODEL_NAME}:generateContent?key={GEMINI_API_KEY}"
)


class AnswerMode(str, Enum):
    detailed = "detailed"
    concise = "concise"
    bullet = "bullet"


def _build_style_instruction(mode: AnswerMode) -> str:
    if mode == AnswerMode.concise:
        return "Answer briefly in 3–5 sentences."
    if mode == AnswerMode.bullet:
        return "Answer using clear bullet points only."
    return "Answer in a detailed, well‑structured explanation."


def _call_gemini(prompt: str) -> str:
    if not GEMINI_API_KEY:
        return "Gemini error: GEMINI_API_KEY is not set in the backend .env."

    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt}
                ]
            }
        ]
    }

    try:
        res = requests.post(GEMINI_URL, json=payload, headers={
            "Content-Type": "application/json"
        })

        if res.status_code != 200:
            return f"Gemini API error: {res.status_code} - {res.text}"

        data = res.json()
        # standard generateContent shape
        return data["candidates"][0]["content"]["parts"][0]["text"]

    except Exception as e:
        return f"Gemini exception: {str(e)}"


# -----------------------------------------------------------
#  Used by /api/chat and RAG helpers (single / multi PDF)
# -----------------------------------------------------------

def run_pdf_qa_llm(
    question: str,
    context: str,
    mode: AnswerMode = AnswerMode.detailed,
) -> str:
    """
    Main QA helper used by:
      - /api/chat
      - answer_question_from_pdf / answer_question_across_pdfs
    Now powered by Gemini 2.5 Pro.
    """
    style = _build_style_instruction(mode)

    prompt = f"""
You are Kuro, an AI assistant that answers questions about PDF documents.

CONTEXT FROM PDF(S):
{context or "(no extra context provided)"}

USER QUESTION:
{question}

INSTRUCTIONS:
- Use the context when it is relevant.
- If the answer is not in the context, say you are not sure and answer from general knowledge.
- {style}
"""

    return _call_gemini(prompt)


# -----------------------------------------------------------
#  Used by /api/pdf/analyse (summary / mcq / flashcards...)
# -----------------------------------------------------------

def run_summarize_llm(
    task_instruction: str,
    full_text: str,
    mode: AnswerMode = AnswerMode.detailed,
) -> str:
    """
    Used for summary / key points / MCQs / flashcards etc.
    """
    style = _build_style_instruction(mode)

    prompt = f"""
You are Kuro, an AI assistant helping with PDF study material.

TASK:
{task_instruction}

DOCUMENT TEXT:
{full_text}

INSTRUCTIONS:
- Follow the task exactly.
- {style}
"""

    return _call_gemini(prompt)
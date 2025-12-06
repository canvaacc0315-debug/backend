import os
import uuid
import json
from typing import List, Optional, Dict, Any
from datetime import datetime

import fitz  # PyMuPDF
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Form, Body, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from pdf_utils import (
    save_uploaded_pdf,           # kept for backwards‑compat
    edit_pdf_add_text,
    edit_pdf_add_image,
    create_custom_pdf_with_images,
)
from rag_utils import (
    index_pdf,
    answer_question_from_pdf,
    answer_question_across_pdfs,
)
from ai_utils import AnswerMode, run_pdf_qa_llm  # 👈 make sure this exists
from auth import get_current_user
from chat_history_utils import (
    list_conversations,
    get_conversation,
    save_conversation,
    delete_conversation,
    clear_all_conversations,
)

load_dotenv()

# ---------- DIRECTORIES ----------
BASE_UPLOAD_DIR = os.path.join("data", "uploads")
BASE_GENERATED_DIR = os.path.join("data", "generated")

os.makedirs(BASE_UPLOAD_DIR, exist_ok=True)
os.makedirs(BASE_GENERATED_DIR, exist_ok=True)

BASE_SHARE_DIR = os.path.join("data", "shares")
os.makedirs(BASE_SHARE_DIR, exist_ok=True)

# ---------- CREATE APP ----------
app = FastAPI(title="PDF Genie API")

# CORS (frontend localhost)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "*",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"message": "PDF Genie API is running"}


# ============================================================
#  ORIGINAL PDF VIEW
# ============================================================

@app.get("/api/pdf/view/{pdf_id}")
async def view_original_pdf(
    pdf_id: str,
    user_id: str = Depends(get_current_user),
):
    """
    Return the original uploaded PDF for the logged‑in user.
    Path: data/uploads/{user_id}/{pdf_id}.pdf
    """
    user_upload_dir = os.path.join(BASE_UPLOAD_DIR, user_id)
    pdf_path = os.path.join(user_upload_dir, f"{pdf_id}.pdf")

    print("Trying to serve:", pdf_path)

    if not os.path.isfile(pdf_path):
        return JSONResponse(
            status_code=404,
            content={"error": f"PDF not found at {pdf_path}"},
        )

    return FileResponse(
        path=pdf_path,
        media_type="application/pdf",
        filename=f"{pdf_id}.pdf",
    )


# ============================================================
#  PDF UPLOAD & INDEX
# ============================================================

@app.post("/api/pdf/upload")
async def upload_pdf(
    files: List[UploadFile] = File(...),
    user_id: str = Depends(get_current_user),
):
    """
    Save PDFs under data/uploads/{user_id}/{pdf_id}.pdf
    and index them for RAG.
    """
    user_upload_dir = os.path.join(BASE_UPLOAD_DIR, user_id)
    os.makedirs(user_upload_dir, exist_ok=True)

    pdf_infos = []

    for file in files:
        pdf_id = str(uuid.uuid4())
        pdf_path = os.path.join(user_upload_dir, f"{pdf_id}.pdf")

        # Save uploaded file
        with open(pdf_path, "wb") as f:
            f.write(await file.read())

        # Index for chat
        index_pdf(pdf_id, pdf_path)

        pdf_infos.append(
            {
                "pdf_id": pdf_id,
                "filename": file.filename,
            }
        )

    return {"pdfs": pdf_infos}


# ============================================================
#  CHAT WITH SINGLE / MULTI PDF
# ============================================================

@app.post("/api/pdf/chat")
async def chat_with_pdf(
    pdf_id: str = Form(...),
    query: str = Form(...),
    mode: AnswerMode = Form("detailed"),
    max_chunks: int = Form(5),
    user_id: str = Depends(get_current_user),
):
    """
    Chat about ONE specific PDF.
    """
    answer, sources = answer_question_from_pdf(
        pdf_id,
        query,
        max_chunks=max_chunks,
        mode=mode,
    )
    return {"answer": answer, "sources": sources}


@app.post("/api/pdf/chat-multi")
async def chat_with_multiple_pdfs(
    pdf_ids: List[str] = Body(...),
    query: str = Body(...),
    mode: AnswerMode = Body("detailed"),
    user_id: str = Depends(get_current_user),
):
    """
    Chat across MANY PDFs (already indexed).
    """
    answer, sources = answer_question_across_pdfs(pdf_ids, query, mode=mode)
    return {"answer": answer, "sources": sources}


# ============================================================
#  GENERIC CHAT ENDPOINT USED BY FRONTEND: /api/chat
# ============================================================

@app.post("/api/chat")
async def api_chat(
    payload: Dict[str, Any] = Body(...),
    user_id: str = Depends(get_current_user),
):
    """
    Generic chat endpoint used by the Kuro workspace UI.

    Frontend sends JSON:
      { "question": "...", "mode": "detailed" | "concise" | "bullet", "pdfId": "..." }

    If pdfId is provided -> answer_question_from_pdf (RAG).
    If not -> plain LLM chat with no PDF context.
    """
    question = (payload.get("question") or "").strip()
    mode_raw = payload.get("mode", "detailed")
    pdf_id = payload.get("pdfId")  # <- comes from frontend
    max_chunks = int(payload.get("max_chunks", 5))

    if not question:
        return JSONResponse(
            status_code=400,
            content={"error": "Missing 'question' in request body."},
        )

    # Safely coerce to AnswerMode enum
    try:
        mode = AnswerMode(mode_raw)
    except Exception:
        mode = AnswerMode.detailed

    # 🔍 Case 1: user selected a single PDF -> use RAG
    if pdf_id:
        try:
            # 👇 NEW: make sure a PDF file exists and (re)build index if needed
            user_upload_dir = os.path.join(BASE_UPLOAD_DIR, user_id)
            pdf_path = os.path.join(user_upload_dir, f"{pdf_id}.pdf")

            if not os.path.exists(pdf_path):
                return JSONResponse(
                    status_code=404,
                    content={"error": f"PDF file not found for id '{pdf_id}'"},
                )

            # (Re)build index on demand – safe even if it already exists
            try:
                index_pdf(pdf_id, pdf_path)
            except Exception as e:
                # don't crash on indexing errors; just surface a clear message
                return JSONResponse(
                    status_code=500,
                    content={"error": f"Indexing error for PDF '{pdf_id}': {str(e)}"},
                )

            # Now answer using the (re)built index
            answer, sources = answer_question_from_pdf(
                pdf_id=pdf_id,
                query=question,
                max_chunks=max_chunks,
                mode=mode,
            )
            return {"answer": answer, "sources": sources}
        except Exception as e:
            return JSONResponse(
                status_code=500,
                content={"error": f"PDF QA error: {str(e)}"},
            )

    # 💬 Case 2: no PDF selected -> generic chat
    try:
        answer = run_pdf_qa_llm(
            question=question,
            context="",   # no PDF context
            mode=mode,
        )
        return {"answer": answer, "sources": []}
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"AI backend error: {str(e)}"},
        )
# ============================================================
#  ANALYSIS (SUMMARY / MCQ / FLASHCARDS etc.)
# ============================================================

@app.post("/api/pdf/analyse")
async def analyse_pdf(
    pdf_id: str = Form(...),
    task: str = Form(...),
    mode_raw: str = Form("detailed"),   # 👈 accept raw string instead of AnswerMode
    user_id: str = Depends(get_current_user),
):
    """
    Read the user's PDF and run summary/MCQ/etc via LLM.
    """
    user_upload_dir = os.path.join(BASE_UPLOAD_DIR, user_id)
    pdf_path = os.path.join(user_upload_dir, f"{pdf_id}.pdf")

    if not os.path.exists(pdf_path):
        return JSONResponse(status_code=404, content={"error": "PDF not found"})

    # 👇 NEW: safely coerce to AnswerMode enum
    try:
        mode = AnswerMode(mode_raw)
    except Exception:
        mode = AnswerMode.detailed

    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text() + "\n\n"
    doc.close()

    from ai_utils import run_summarize_llm

    prompt_map = {
        "summary": "Create a clear, structured summary of this document.",
        "key_points": "Extract the key points as bullet points.",
        "definitions": "Extract important terms with their definitions.",
        "flashcards": "Create 10 Q&A flashcards in simple language.",
        "mcq": "Create 10 multiple-choice questions with 4 options and mark the correct answer.",
        "study_guide": "Create a short study guide for revision.",
    }
    task_instruction = prompt_map.get(task, task)

    result = run_summarize_llm(task_instruction, text, mode=mode)
    return {"result": result}

# ============================================================
#  SHARE CHAT CONVERSATION
# ============================================================

@app.post("/api/chat/share")
async def create_share_link(
    payload: Dict[str, Any] = Body(...),
    user_id: str = Depends(get_current_user),
):
    """
    Save a conversation to disk and return a short slug.
    Frontend will build the final URL as `${window.location.origin}/share/{slug}`.
    """
    conversation = payload.get("conversation", [])
    if not isinstance(conversation, list) or not conversation:
        return JSONResponse(
            status_code=400,
            content={"error": "Missing or empty 'conversation' array."},
        )

    slug = uuid.uuid4().hex[:8]  # short id like "a1b2c3d4"
    share_path = os.path.join(BASE_SHARE_DIR, f"{slug}.json")

    save_obj = {
        "slug": slug,
        "user_id": user_id,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "conversation": conversation,
    }

    with open(share_path, "w", encoding="utf-8") as f:
        json.dump(save_obj, f, ensure_ascii=False, indent=2)

    # you can later create a GET /api/chat/share/{slug} to read this
    return {"slug": slug}

# ============================================================
#  EDIT PDF: ADD TEXT
# ============================================================

@app.post("/api/pdf/edit/add-text")
async def api_edit_pdf_add_text(
    pdf_id: str = Form(...),
    page_number: int = Form(...),
    text: str = Form(...),
    x: float = Form(50),
    y: float = Form(50),
    font_size: int = Form(12),
    user_id: str = Depends(get_current_user),
):
    user_upload_dir = os.path.join(BASE_UPLOAD_DIR, user_id)
    input_path = os.path.join(user_upload_dir, f"{pdf_id}.pdf")
    if not os.path.exists(input_path):
        return JSONResponse(status_code=404, content={"error": "PDF not found"})

    output_filename = f"{pdf_id}_edited_{uuid.uuid4().hex}.pdf"
    output_path = os.path.join(BASE_GENERATED_DIR, output_filename)

    try:
        edit_pdf_add_text(
            input_pdf_path=input_path,
            output_pdf_path=output_path,
            page_number=page_number,
            text=text,
            x=x,
            y=y,
            font_size=font_size,
        )
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

    return {"status": "ok", "edited_pdf": output_filename}


# ============================================================
#  EDIT PDF: ADD IMAGE
# ============================================================

@app.post("/api/pdf/edit/add-image")
async def api_edit_pdf_add_image(
    pdf_id: str = Form(...),
    page_number: int = Form(...),
    x: float = Form(50),
    y: float = Form(50),
    width: Optional[float] = Form(None),
    height: Optional[float] = Form(None),
    image: UploadFile = File(...),
    user_id: str = Depends(get_current_user),
):
    user_upload_dir = os.path.join(BASE_UPLOAD_DIR, user_id)
    input_path = os.path.join(user_upload_dir, f"{pdf_id}.pdf")
    if not os.path.exists(input_path):
        return JSONResponse(status_code=404, content={"error": "PDF not found"})

    img_ext = os.path.splitext(image.filename)[1] or ".png"
    img_name = f"{uuid.uuid4().hex}{img_ext}"
    img_path = os.path.join(BASE_GENERATED_DIR, img_name)

    with open(img_path, "wb") as f:
        f.write(await image.read())

    output_filename = f"{pdf_id}_img_{uuid.uuid4().hex}.pdf"
    output_path = os.path.join(BASE_GENERATED_DIR, output_filename)

    try:
        edit_pdf_add_image(
            input_pdf_path=input_path,
            output_pdf_path=output_path,
            page_number=page_number,
            image_path=img_path,
            x=x,
            y=y,
            width=width,
            height=height,
        )
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

    return {"status": "ok", "edited_pdf": output_filename}


# ============================================================
#  CREATE CUSTOM PDF
# ============================================================

@app.post("/api/pdf/create")
async def api_create_custom_pdf(
    title: str = Form("Custom PDF"),
    body_text: str = Form(""),
    images: List[UploadFile] = File([]),
    user_id: str = Depends(get_current_user),
):
    img_paths: List[str] = []

    for image in images:
        img_ext = os.path.splitext(image.filename)[1] or ".png"
        img_name = f"{uuid.uuid4().hex}{img_ext}"
        img_path = os.path.join(BASE_GENERATED_DIR, img_name)
        with open(img_path, "wb") as f:
            f.write(await image.read())
        img_paths.append(img_path)

    pdf_name = f"custom_{uuid.uuid4().hex}.pdf"
    pdf_path = os.path.join(BASE_GENERATED_DIR, pdf_name)

    create_custom_pdf_with_images(
        output_pdf_path=pdf_path,
        title=title,
        body_text=body_text,
        image_paths=img_paths,
    )

    return {"status": "ok", "pdf": pdf_name}


# ============================================================
#  DOWNLOAD GENERATED PDF
# ============================================================

@app.get("/api/pdf/download/{filename}")
async def download_pdf(filename: str):
    file_path = os.path.join(BASE_GENERATED_DIR, filename)
    if not os.path.exists(file_path):
        return JSONResponse(status_code=404, content={"error": "File not found"})

    return FileResponse(
        path=file_path,
        media_type="application/pdf",
        filename=filename,
    )


# ============================================================
#  OCR / TEXT EXTRACTION
# ============================================================

@app.post("/api/pdf/ocr")
async def ocr_pdf(
    pdf_id: str = Form(...),
    user_id: str = Depends(get_current_user),
):
    """
    Very simple 'OCR' / text extraction endpoint.
    For now it uses PyMuPDF's get_text() on each page.
    """
    user_upload_dir = os.path.join(BASE_UPLOAD_DIR, user_id)
    pdf_path = os.path.join(user_upload_dir, f"{pdf_id}.pdf")

    if not os.path.exists(pdf_path):
        return JSONResponse(status_code=404, content={"error": "PDF not found"})

    doc = fitz.open(pdf_path)
    extracted = ""
    for page in doc:
        extracted += page.get_text("text") + "\n\n"
# List history
@app.get("/api/chat/history")
async def list_chat_history(user_id: str = Depends(get_current_user)):
    items = list_conversations(user_id)
    return {"items": items}

# Get single conversation
@app.get("/api/chat/history/{conv_id}")
async def get_chat_conversation(conv_id: str, user_id: str = Depends(get_current_user)):
    convo = get_conversation(user_id, conv_id)
    if convo is None:
        return JSONResponse(status_code=404, content={"error": "Conversation not found"})
    return convo

# Save a conversation
class SaveConversationPayload(BaseModel):
    title: str
    messages: list  # array of {id, role, content, timestamp}

@app.post("/api/chat/history")
async def save_chat_history(payload: SaveConversationPayload, user_id: str = Depends(get_current_user)):
    conv_id = save_conversation(user_id, payload)
    return {"id": conv_id}

# Delete One
@app.delete("/api/chat/history/{conv_id}")
async def delete_chat_history(conv_id: str, user_id: str = Depends(get_current_user)):
    ok = delete_conversation(user_id, conv_id)
    if not ok:
        return JSONResponse(status_code=404, content={"error": "Conversation not found"})
    return {"status": "ok"}

# Clear All
@app.delete("/api/chat/history")
async def clear_all_chat_history(user_id: str = Depends(get_current_user)):
    clear_all_conversations(user_id)
    return {"status": "ok"}


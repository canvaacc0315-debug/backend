import os
import pickle
from typing import List, Tuple
from ai_utils import run_pdf_qa_llm, AnswerMode

import fitz  # PyMuPDF
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Load model once
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

INDEX_DIR = os.path.join("data", "indexes")
os.makedirs(INDEX_DIR, exist_ok=True)


def _get_index_paths(pdf_id: str) -> Tuple[str, str]:
    index_path = os.path.join(INDEX_DIR, f"{pdf_id}_index.faiss")
    meta_path = os.path.join(INDEX_DIR, f"{pdf_id}_meta.pkl")
    return index_path, meta_path


# Minimum character threshold — pages with fewer chars trigger OCR fallback
OCR_FALLBACK_THRESHOLD = 50


def index_pdf(pdf_id: str, pdf_path: str) -> None:
    """
    Extract text from PDF, chunk it, embed it, and store FAISS index + metadata.
    If a page has little or no native text, OCR is used as a fallback.
    """
    # Lazy import — only loaded if OCR is actually needed
    from ocr_utils import run_ocr_on_page

    doc = fitz.open(pdf_path)

    chunks: List[str] = []
    metadatas: List[dict] = []
    ocr_pages = 0

    for page_number in range(len(doc)):
        page = doc[page_number]
        text = page.get_text().strip()
        source = "native"

        # --- OCR Fallback: if native text is missing or too short ---
        if len(text) < OCR_FALLBACK_THRESHOLD:
            print(f"  Page {page_number + 1}: native text too short "
                  f"({len(text)} chars), running OCR fallback...")
            ocr_text = run_ocr_on_page(pdf_path, page_number)
            if ocr_text:
                text = ocr_text
                source = "ocr"
                ocr_pages += 1
            else:
                print(f"  Page {page_number + 1}: OCR also returned nothing, skipping.")
                continue

        # Naive chunking: split by lines, group to ~400+ characters
        lines = text.split("\n")
        buffer: List[str] = []

        for line in lines:
            buffer.append(line)
            if len(" ".join(buffer)) > 400:
                chunk_text = " ".join(buffer)
                chunks.append(chunk_text)
                metadatas.append({
                    "page_number": page_number,
                    "text": chunk_text,
                    "source": source,
                })
                buffer = []

        if buffer:
            chunk_text = " ".join(buffer)
            chunks.append(chunk_text)
            metadatas.append({
                "page_number": page_number,
                "text": chunk_text,
                "source": source,
            })

    doc.close()

    if not chunks:
        print("No text found in PDF for indexing (native + OCR both failed).")
        return

    # Create embeddings
    embeddings = model.encode(chunks, convert_to_numpy=True)
    d = embeddings.shape[1]

    index = faiss.IndexFlatL2(d)
    index.add(embeddings)

    index_path, meta_path = _get_index_paths(pdf_id)
    faiss.write_index(index, index_path)
    with open(meta_path, "wb") as f:
        pickle.dump(metadatas, f)

    print(f"Indexed PDF {pdf_id}: {len(chunks)} chunks "
          f"({ocr_pages} pages used OCR fallback)")


def answer_question_from_pdf(
    pdf_id: str,
    query: str,
    max_chunks: int = 5,
    mode: AnswerMode = "detailed",
):
    index_path, meta_path = _get_index_paths(pdf_id)

    if not (os.path.exists(index_path) and os.path.exists(meta_path)):
        return "No index found for this PDF. Upload it again to build index.", []

    index = faiss.read_index(index_path)
    with open(meta_path, "rb") as f:
        metadatas = pickle.load(f)

    query_embedding = model.encode([query], convert_to_numpy=True)
    D, I = index.search(query_embedding, max_chunks)

    retrieved = []
    for idx in I[0]:
        meta = metadatas[idx]
        retrieved.append(meta)

    # Use top chunks as context for LLM
    context_chunks = [r["text"] for r in retrieved]
    answer = run_pdf_qa_llm(query, context_chunks, mode=mode)

    sources = [
        {
            "page_number": r["page_number"],
            "snippet": r["text"][:300],
        }
        for r in retrieved
    ]

    return answer, sources

def answer_question_across_pdfs(
    pdf_ids: List[str],
    query: str,
    max_chunks_per_pdf: int = 3,
    mode: AnswerMode = "detailed",
):
    all_chunks = []
    all_sources = []

    for pdf_id in pdf_ids:
        index_path, meta_path = _get_index_paths(pdf_id)
        if not (os.path.exists(index_path) and os.path.exists(meta_path)):
            continue

        index = faiss.read_index(index_path)
        with open(meta_path, "rb") as f:
            metadatas = pickle.load(f)

        query_embedding = model.encode([query], convert_to_numpy=True)
        D, I = index.search(query_embedding, max_chunks_per_pdf)

        for idx in I[0]:
            meta = metadatas[idx]
            all_chunks.append(meta["text"])
            all_sources.append({
                "pdf_id": pdf_id,
                "page_number": meta["page_number"],
                "snippet": meta["text"][:300],
            })

    if not all_chunks:
        return "I couldn't find relevant content in these PDFs.", []

    answer = run_pdf_qa_llm(query, all_chunks, mode=mode)
    return answer, all_sources


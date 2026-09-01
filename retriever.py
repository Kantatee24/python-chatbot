import re
import pandas as pd
from pathlib import Path

CHUNK_SIZE = 500
CHUNK_OVERLAP = 80
PDF_TOP_K = 2
EXCEL_TOP_K = 3


def load_chunks(txt_path: str) -> list[str]:
    text = Path(txt_path).read_text(encoding="utf-8")
    text = re.sub(r"\n{3,}", "\n\n", text)
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i : i + CHUNK_SIZE])
        chunks.append(chunk)
        i += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def load_qa(xlsx_path: str) -> list[dict]:
    df = pd.read_excel(xlsx_path).fillna("")
    rows = []
    for _, row in df.iterrows():
        rows.append({
            "topic": f"{row['หัวข้อหลัก']} {row['หัวข้อย่อย']} {row['ย่อยรอง']}",
            "question": str(row["คำถาม (แก้ไขใหม่)"]),
            "answer": str(row["คำตอบ"]),
            "keywords": str(row["keyword ตรง"]),
        })
    return rows


def _score_words(query: str, text: str) -> int:
    q = set(re.findall(r"\w+", query.lower()))
    t = set(re.findall(r"\w+", text.lower()))
    return len(q & t)


def search_qa(query: str, qa_rows: list[dict], top_k: int = EXCEL_TOP_K) -> str:
    scores = []
    for i, row in enumerate(qa_rows):
        score = (
            _score_words(query, row["question"]) * 3
            + _score_words(query, row["keywords"]) * 2
            + _score_words(query, row["topic"])
        )
        scores.append((score, i))
    scores.sort(reverse=True)
    best = [(s, i) for s, i in scores[:top_k] if s > 0]
    if not best:
        return ""
    parts = []
    for _, i in best:
        r = qa_rows[i]
        parts.append(f"Q: {r['question']}\nA: {r['answer']}")
    return "\n\n".join(parts)


def search_pdf(query: str, chunks: list[str], top_k: int = PDF_TOP_K) -> str:
    scores = []
    for i, chunk in enumerate(chunks):
        score = _score_words(query, chunk)
        scores.append((score, i))
    scores.sort(reverse=True)
    top_chunks = [chunks[i] for s, i in scores[:top_k] if s > 0]
    if not top_chunks:
        return chunks[0]
    return "\n\n---\n\n".join(top_chunks)


def search(query: str, chunks: list[str], qa_rows: list[dict]) -> str:
    qa_result = search_qa(query, qa_rows)
    pdf_result = search_pdf(query, chunks)
    parts = []
    if qa_result:
        parts.append(f"[จาก Q&A Excel]\n{qa_result}")
    if pdf_result:
        parts.append(f"[จากหนังสือ PDF]\n{pdf_result}")
    return "\n\n===\n\n".join(parts)

import re
import numpy as np
import pandas as pd
from pathlib import Path

CHUNK_SIZE = 500
CHUNK_OVERLAP = 80
PDF_TOP_K = 4
EXCEL_TOP_K = 5

# คำถามทั่วไปที่ไม่ช่วย discriminate — ตัดออกจาก score
THAI_STOPWORDS = {
    "เขียน", "ยังไง", "คืออะไร", "ใช้ยังไง", "อะไร", "ทำไม", "อย่างไร",
    "หน่อย", "ด้วย", "ครับ", "ค่ะ", "นะ", "บ้าง", "ได้", "ให้", "แบบ",
    "ตัวอย่าง", "วิธี", "การ", "ของ", "ที่", "เป็น", "มี", "ใน",
    "กับ", "และ", "หรือ", "แต่", "จะ", "ก็", "แล้ว", "จาก", "โดย",
}

PY_KEYWORDS = {
    "if", "else", "elif", "for", "while", "def", "class", "return",
    "import", "from", "try", "except", "with", "as", "and", "or", "not",
    "in", "is", "lambda", "pass", "break", "continue", "yield", "global",
    "print", "list", "dict", "tuple", "set", "range", "len", "type",
    "str", "int", "float", "bool", "none", "true", "false", "input",
    "open", "append", "remove", "sort", "map", "filter", "zip",
}


def clean_thai(text: str) -> str:
    """แก้ encoding artifact จาก PDF: [พยัญชนะ][space][สระา] → [พยัญชนะ][สระำ]"""
    # fix sara am (ำ) ที่ถูก encode เป็น space + sara aa (า)
    text = re.sub(r'([ก-ฮ]) า', r'\1ำ', text)
    # fix vowel marks อื่นที่หลุด space
    text = re.sub(r'([ก-ฮ]) ([ัิ-ู็-๎])', r'\1\2', text)
    return text


def load_chunks(md_path: str) -> list[str]:
    text = Path(md_path).read_text(encoding="utf-8")
    text = clean_thai(text)                          # fix encoding ก่อน
    raw_sections = re.split(r'\n(?=#{1,3} )', text)
    chunks = []
    for section in raw_sections:
        section = section.strip()
        if not section:
            continue
        words = section.split()
        if len(words) <= CHUNK_SIZE:
            chunks.append(section)
        else:
            # เก็บ heading ไว้ใน sub-chunk ด้วย
            heading_line = section.split('\n')[0]
            i = 0
            while i < len(words):
                sub = " ".join(words[i: i + CHUNK_SIZE])
                if i > 0:
                    sub = heading_line + "\n" + sub   # แนบ heading ทุก sub-chunk
                chunks.append(sub)
                i += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def build_heading_index(chunks: list[str]) -> list[dict]:
    """สร้าง index ของ heading แต่ละ chunk สำหรับค้นหา precision สูง"""
    index = []
    for i, chunk in enumerate(chunks):
        first_line = chunk.split('\n')[0].strip()
        m = re.match(r'^#{1,3}\s+(.+)', first_line)
        if m:
            heading = re.sub(r'\*+|_+', '', m.group(1)).strip()
            index.append({"heading": heading, "chunk_idx": i})
    return index


def load_embeddings(path: str) -> np.ndarray | None:
    try:
        return np.load(path)
    except Exception:
        return None


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


def _score(query: str, text: str) -> float:
    q_words = set(re.findall(r"\w+", query.lower())) - THAI_STOPWORDS
    t_words = set(re.findall(r"\w+", text.lower()))
    score = float(len(q_words & t_words))

    # boost ถ้า heading ของ chunk ตรงกับ query
    heading = re.match(r'^#{1,3}\s+(.+)', text)
    if heading:
        h_words = set(re.findall(r"\w+", heading.group(1).lower()))
        score += len(q_words & h_words) * 4

    # boost Python keywords
    py_in_q = q_words & PY_KEYWORDS
    py_in_t = t_words & PY_KEYWORDS
    score += len(py_in_q & py_in_t) * 3

    return score


def search_qa(query: str, qa_rows: list[dict], top_k: int = EXCEL_TOP_K) -> str:
    clean_q = clean_thai(query)
    scores = []
    for i, row in enumerate(qa_rows):
        score = (
            _score(clean_q, row["question"]) * 3
            + _score(clean_q, row["keywords"]) * 2
            + _score(clean_q, row["topic"])
        )
        scores.append((score, i))
    scores.sort(reverse=True)
    best = [(s, i) for s, i in scores[:top_k] if s > 0]
    if not best:
        return ""
    parts = [f"Q: {qa_rows[i]['question']}\nA: {qa_rows[i]['answer']}" for _, i in best]
    return "\n\n".join(parts)


def _semantic_search(query_vec: np.ndarray, embeddings: np.ndarray,
                     chunks: list[str], top_k: int) -> list[str]:
    norms = np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_vec)
    sims = np.dot(embeddings, query_vec) / (norms + 1e-9)
    top_idx = np.argsort(sims)[::-1][:top_k]
    return [chunks[i] for i in top_idx]


def search_pdf(query: str, chunks: list[str], heading_index: list[dict],
               top_k: int = PDF_TOP_K,
               embeddings: np.ndarray | None = None,
               embed_fn=None) -> str:

    if embeddings is not None and embed_fn is not None and len(embeddings) == len(chunks):
        try:
            query_vec = embed_fn(clean_thai(query))
            result = _semantic_search(query_vec, embeddings, chunks, top_k)
            return "\n\n---\n\n".join(result)
        except Exception:
            pass  # fall through to keyword search

    # keyword fallback
    clean_q = clean_thai(query)
    h_scores = [(round(_score(clean_q, h["heading"]), 2), h["chunk_idx"])
                for h in heading_index]
    h_scores.sort(reverse=True)
    seen = set()
    heading_chunks = []
    for s, i in h_scores:
        if s > 0 and i not in seen:
            heading_chunks.append(chunks[i])
            seen.add(i)
        if len(heading_chunks) >= 2:
            break

    c_scores = [(round(_score(clean_q, c), 2), i) for i, c in enumerate(chunks)]
    c_scores.sort(reverse=True)
    extra = [chunks[i] for s, i in c_scores if s > 0 and i not in seen][: top_k - len(heading_chunks)]
    result = heading_chunks + extra
    if not result:
        result = [chunks[i] for _, i in c_scores[:2]]
    return "\n\n---\n\n".join(result)


def search(query: str, chunks: list[str], heading_index: list[dict],
           qa_rows: list[dict],
           embeddings: np.ndarray | None = None,
           embed_fn=None) -> str:
    qa_result = search_qa(query, qa_rows)
    pdf_result = search_pdf(query, chunks, heading_index,
                            embeddings=embeddings, embed_fn=embed_fn)
    parts = []
    if qa_result:
        parts.append(f"[จาก Q&A Excel]\n{qa_result}")
    if pdf_result:
        parts.append(f"[จากหนังสือ Markdown]\n{pdf_result}")
    return "\n\n===\n\n".join(parts)

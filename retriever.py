import re
import pandas as pd
from pathlib import Path

CHUNK_SIZE = 500
CHUNK_OVERLAP = 80
PDF_TOP_K = 4
EXCEL_TOP_K = 3

# Python keywords — ถ้า query มีคำเหล่านี้ให้ boost คะแนน
PY_KEYWORDS = {
    "if", "else", "elif", "for", "while", "def", "class", "return",
    "import", "from", "try", "except", "with", "as", "and", "or", "not",
    "in", "is", "lambda", "pass", "break", "continue", "yield", "global",
    "print", "list", "dict", "tuple", "set", "range", "len", "type",
    "str", "int", "float", "bool", "none", "true", "false", "input",
    "open", "file", "append", "remove", "sort", "map", "filter", "zip",
}


def load_chunks(md_path: str) -> list[str]:
    """Split Markdown by headings first, then by word count if section too long."""
    text = Path(md_path).read_text(encoding="utf-8")
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
            i = 0
            while i < len(words):
                chunks.append(" ".join(words[i : i + CHUNK_SIZE]))
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


def _score(query: str, text: str) -> float:
    q_words = set(re.findall(r"\w+", query.lower()))
    t_words = set(re.findall(r"\w+", text.lower()))

    score = float(len(q_words & t_words))

    # boost ถ้า heading ของ chunk ตรงกับ query
    heading = re.match(r'^#{1,3}\s+(.+)', text)
    if heading:
        h_words = set(re.findall(r"\w+", heading.group(1).lower()))
        score += len(q_words & h_words) * 4

    # boost สำหรับ Python keywords ที่ตรงกัน
    py_in_query = q_words & PY_KEYWORDS
    py_in_text = t_words & PY_KEYWORDS
    score += len(py_in_query & py_in_text) * 3

    return score


def search_qa(query: str, qa_rows: list[dict], top_k: int = EXCEL_TOP_K) -> str:
    scores = []
    for i, row in enumerate(qa_rows):
        score = (
            _score(query, row["question"]) * 3
            + _score(query, row["keywords"]) * 2
            + _score(query, row["topic"])
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
    scores = [(round(_score(query, c), 2), i) for i, c in enumerate(chunks)]
    scores.sort(reverse=True)

    # เอา top_k ที่ score > 0 ก่อน ถ้าไม่มีเลยให้เอา top 2 มาเลย
    top = [chunks[i] for s, i in scores[:top_k] if s > 0]
    if not top:
        top = [chunks[i] for _, i in scores[:2]]

    return "\n\n---\n\n".join(top)


def search(query: str, chunks: list[str], qa_rows: list[dict]) -> str:
    qa_result = search_qa(query, qa_rows)
    pdf_result = search_pdf(query, chunks)
    parts = []
    if qa_result:
        parts.append(f"[จาก Q&A Excel]\n{qa_result}")
    if pdf_result:
        parts.append(f"[จากหนังสือ Markdown]\n{pdf_result}")
    return "\n\n===\n\n".join(parts)

"""
รัน script นี้ครั้งเดียวเพื่อสร้าง embeddings.npy
  python build_embeddings.py
สามารถรันซ้ำได้ถ้าหยุดกลางทาง — จะต่อจากที่ค้างไว้
"""
import os, time
import numpy as np
from dotenv import load_dotenv
from google import genai
from retriever import load_chunks

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

BASE_DIR = os.path.dirname(__file__)
MD_PATH  = os.path.join(BASE_DIR, "python_data.md")
OUT_PATH = os.path.join(BASE_DIR, "embeddings.npy")
TMP_PATH = os.path.join(BASE_DIR, "embeddings_tmp.npy")
MODEL    = "gemini-embedding-001"
DELAY    = 1.2

chunks = load_chunks(MD_PATH)
total  = len(chunks)
print(f"Chunks: {total}")

# โหลด checkpoint ถ้ามี
if os.path.exists(TMP_PATH):
    all_vecs = list(np.load(TMP_PATH))
    start = len(all_vecs)
    print(f"Resume from {start}/{total}")
else:
    all_vecs = []
    start = 0

for i in range(start, total):
    delay = DELAY
    for attempt in range(6):
        try:
            result = client.models.embed_content(model=MODEL, contents=chunks[i][:2000])
            all_vecs.append(result.embeddings[0].values)
            break
        except Exception as e:
            print(f"  [{i+1}] retry {attempt+1}: {e}")
            time.sleep(delay)
            delay = min(delay * 2, 120)
    else:
        print(f"  [{i+1}] SKIP หลัง 6 ครั้ง")
        all_vecs.append([0.0] * 3072)

    if (i + 1) % 20 == 0 or i == total - 1:
        np.save(TMP_PATH, np.array(all_vecs, dtype=np.float32))
        print(f"  checkpoint {i+1}/{total}")

    time.sleep(DELAY)

arr = np.array(all_vecs, dtype=np.float32)
np.save(OUT_PATH, arr)
if os.path.exists(TMP_PATH):
    os.remove(TMP_PATH)
print(f"Done → embeddings.npy  shape={arr.shape}")

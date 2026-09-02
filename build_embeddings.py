"""
รัน script นี้ครั้งเดียวเพื่อสร้าง embeddings.npy
  python build_embeddings.py
"""
import os, time
import numpy as np
from dotenv import load_dotenv
from google import genai
from retriever import load_chunks

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

BASE_DIR = os.path.dirname(__file__)
MD_PATH = os.path.join(BASE_DIR, "python_data.md")
OUT_PATH = os.path.join(BASE_DIR, "embeddings.npy")
MODEL = "gemini-embedding-001"
BATCH = 20

chunks = load_chunks(MD_PATH)
print(f"Chunks: {len(chunks)}")

all_vecs = []
for i, chunk in enumerate(chunks):
    delay = 2
    for attempt in range(5):
        try:
            result = client.models.embed_content(model=MODEL, contents=chunk[:2000])
            all_vecs.append(result.embeddings[0].values)
            if (i + 1) % 20 == 0 or i == len(chunks) - 1:
                print(f"  {i + 1}/{len(chunks)}")
            break
        except Exception as e:
            print(f"  [{i+1}] retry {attempt+1} ({e})")
            time.sleep(delay)
            delay = min(delay * 2, 60)
    time.sleep(1.5)

arr = np.array(all_vecs, dtype=np.float32)
np.save(OUT_PATH, arr)
print(f"Done → embeddings.npy  shape={arr.shape}")

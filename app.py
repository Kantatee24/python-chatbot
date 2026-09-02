import os
import time
import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types, errors
from prompt import PROMPT_PYBOT
from retriever import load_chunks, load_qa, build_heading_index, search

load_dotenv()

api_key = st.secrets.get("GEMINI_API_KEY", "") if hasattr(st, "secrets") else ""
if not api_key:
    api_key = os.getenv("GEMINI_API_KEY", "")
if not api_key or api_key == "your_api_key_here":
    st.error("❌ กรุณาใส่ GEMINI_API_KEY ใน .env หรือ Streamlit Secrets")
    st.stop()

client = genai.Client(api_key=api_key)

SAFETY_SETTINGS = [
    types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE"),
    types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_NONE"),
    types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_NONE"),
    types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"),
]

CHAT_CONFIG = types.GenerateContentConfig(
    temperature=0.0,
    max_output_tokens=2048,
    system_instruction=PROMPT_PYBOT,
    safety_settings=SAFETY_SETTINGS,
)

BASE_DIR = os.path.dirname(__file__)


def expand_query(query: str, chunks: list, heading_index: list, qa_rows: list) -> str:
    """ค้นหาเบื้องต้นก่อน แล้วให้ Gemini ดึง 2-3 keyword จากเนื้อหาในหนังสือ"""
    try:
        initial_context = search(query, chunks, heading_index, qa_rows)
        result = client.models.generate_content(
            model="gemini-3.5-flash-lite",
            contents=(
                f"เนื้อหาจากหนังสือ Python MSU:\n{initial_context[:1500]}\n\n"
                f"คำถาม: {query}\n\n"
                f"จากเนื้อหาข้างต้นเท่านั้น ดึง keywords ที่มีในเนื้อหานั้น 2-3 คำ "
                f"เพื่อช่วยค้นหาส่วนที่เกี่ยวข้องมากขึ้น\n"
                f"ตอบเฉพาะ keywords คั่นด้วย space เท่านั้น ห้ามอธิบาย"
            ),
            config=types.GenerateContentConfig(temperature=0.0, max_output_tokens=20),
        )
        keywords = result.text.strip().replace("\n", " ")
        return f"{query} {keywords}"
    except Exception:
        return query


@st.cache_data(show_spinner="กำลังโหลดหนังสือ Python...")
def get_chunks():
    return load_chunks(os.path.join(BASE_DIR, "python_data.md"))


@st.cache_data(show_spinner="กำลังโหลด Q&A...")
def get_qa():
    return load_qa(os.path.join(BASE_DIR, "python_qa.xlsx"))


@st.cache_data(show_spinner=False)
def get_heading_index():
    return build_heading_index(get_chunks())


def clear_history():
    st.session_state["messages"] = [
        {"role": "model", "content": "สวัสดีครับ! ผม PyBot ผู้ช่วยเรียน Python จากหนังสือ Python MSU ครับ สอบถามเรื่องใดได้เลยครับ 🐍"}
    ]
    st.rerun()


st.set_page_config(page_title="PyBot - ผู้ช่วยเรียน Python", page_icon="🐍")
st.title("🐍 PyBot - ผู้ช่วยเรียน Python")
st.caption("ตอบคำถามจากหนังสือ Python MSU (Suchart)")

with st.sidebar:
    st.header("เมนู")
    if st.button("🗑️ ล้างประวัติการสนทนา"):
        clear_history()
    st.divider()
    st.markdown("**แหล่งข้อมูล:**")
    st.markdown("- Python MSU-Suchart.pdf")
    st.markdown("- ภาษา Python.xlsx (197 Q&A)")

chunks = get_chunks()
qa_rows = get_qa()
heading_index = get_heading_index()

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "model", "content": "สวัสดีครับ! ผม PyBot ผู้ช่วยเรียน Python จากหนังสือ Python MSU ครับ สอบถามเรื่องใดได้เลยครับ 🐍"}
    ]

for msg in st.session_state["messages"]:
    st.chat_message(msg["role"]).write(msg["content"])

if user_input := st.chat_input("ถามเรื่อง Python ได้เลยครับ..."):
    st.session_state["messages"].append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)

    expanded = expand_query(user_input, chunks, heading_index, qa_rows)
    relevant_context = search(expanded, chunks, heading_index, qa_rows)

    history = []
    for msg in st.session_state["messages"][:-1]:
        role = "user" if msg["role"] == "user" else "model"
        history.append(types.Content(role=role, parts=[types.Part(text=msg["content"])]))

    history.insert(0, types.Content(
        role="user",
        parts=[types.Part(text=(
            f"[เนื้อหาจากหนังสือ Python MSU — ใช้เฉพาะเนื้อหานี้ในการตอบเท่านั้น ห้ามเพิ่มเติมจากแหล่งอื่น]\n\n"
            f"{relevant_context}\n\n"
            f"[สิ้นสุดเนื้อหาจากหนังสือ]"
        ))]
    ))
    history.insert(1, types.Content(
        role="model",
        parts=[types.Part(text="รับทราบครับ จะตอบโดยใช้เฉพาะเนื้อหาจากหนังสือที่ให้มาเท่านั้นครับ")]
    ))

    with st.spinner("กำลังคิด..."):
        response = None
        models = ["gemini-3.5-flash-lite", "gemini-3.6-flash"]
        for model in models:
            for attempt in range(3):
                try:
                    chat = client.chats.create(model=model, config=CHAT_CONFIG, history=history)
                    response = chat.send_message(user_input)
                    break
                except errors.ServerError:
                    if attempt < 2:
                        time.sleep(7)
                except Exception as e:
                    st.error(f"❌ Gemini API Error: {e}")
                    break
            if response:
                break
        if not response:
            st.warning("⚠️ เซิร์ฟเวอร์ Gemini ยุ่งมากในขณะนี้ กรุณาลองใหม่สักครู่ครับ")
        if response:
            st.session_state["messages"].append({"role": "model", "content": response.text})
            st.chat_message("model").write(response.text)

import os
import time
import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types, errors
from prompt import PROMPT_PYBOT
from retriever import load_chunks, load_qa, search

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
    temperature=0.2,
    top_p=0.95,
    top_k=64,
    max_output_tokens=1024,
    system_instruction=PROMPT_PYBOT,
    safety_settings=SAFETY_SETTINGS,
)

BASE_DIR = os.path.dirname(__file__)


@st.cache_data(show_spinner="กำลังโหลดหนังสือ Python...")
def get_chunks():
    return load_chunks(os.path.join(BASE_DIR, "python_data.txt"))


@st.cache_data(show_spinner="กำลังโหลด Q&A...")
def get_qa():
    return load_qa(os.path.join(BASE_DIR, "python_qa.xlsx"))


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

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "model", "content": "สวัสดีครับ! ผม PyBot ผู้ช่วยเรียน Python จากหนังสือ Python MSU ครับ สอบถามเรื่องใดได้เลยครับ 🐍"}
    ]

for msg in st.session_state["messages"]:
    st.chat_message(msg["role"]).write(msg["content"])

if user_input := st.chat_input("ถามเรื่อง Python ได้เลยครับ..."):
    st.session_state["messages"].append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)

    relevant_context = search(user_input, chunks, qa_rows)

    history = []
    for msg in st.session_state["messages"][:-1]:
        role = "user" if msg["role"] == "user" else "model"
        history.append(types.Content(role=role, parts=[types.Part(text=msg["content"])]))

    history.insert(0, types.Content(
        role="user",
        parts=[types.Part(text=f"[เนื้อหาที่เกี่ยวข้องจากหนังสือ Python MSU]\n{relevant_context}")]
    ))
    history.insert(1, types.Content(
        role="model",
        parts=[types.Part(text="รับทราบครับ พร้อมตอบคำถามครับ")]
    ))

    with st.spinner("กำลังคิด..."):
        response = None
        for attempt in range(3):
            try:
                chat = client.chats.create(model="gemini-3.5-flash-lite", config=CHAT_CONFIG, history=history)
                response = chat.send_message(user_input)
                break
            except errors.ServerError:
                if attempt < 2:
                    time.sleep(3)
                else:
                    st.warning("⚠️ เซิร์ฟเวอร์ Gemini ยุ่งอยู่ชั่วคราว กรุณาลองใหม่อีกครั้งครับ")
            except Exception as e:
                st.error(f"❌ Gemini API Error: {e}")
                break
        if response:
            st.session_state["messages"].append({"role": "model", "content": response.text})
            st.chat_message("model").write(response.text)

import os
from pathlib import Path
from dotenv import load_dotenv

import streamlit as st
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.embeddings import CacheBackedEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.storage import LocalFileStore
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.callbacks.base import BaseCallbackHandler

# ✅ 설정
STATIC_DIR = Path("./static/schoolviolence")

# ✅ 로그인 확인
if "logged_in" not in st.session_state or not st.session_state.logged_in:
    st.error("🚫 로그인해야 사용가능합니다. Home 으로 가세요.")
    st.markdown("""
        <a href="./" target="_self">🔙 Home으로 가서 로그인하기</a>
    """, unsafe_allow_html=True)
    st.stop()

# ✅ 콜백 핸들러 (스트리밍)
class ChatCallbackHandler(BaseCallbackHandler):
    message = ""
    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()
    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai")
    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)

# ✅ LLM 설정
llm = ChatOpenAI(
    temperature=0.1,
    streaming=True,
    callbacks=[ChatCallbackHandler()],
)

# ✅ 문서 로딩 및 벡터화
@st.cache_resource(show_spinner="문서 로딩중. 잠시만 기다리세요...")
def load_all_documents():
    all_docs = []
    for file_path in STATIC_DIR.glob("*.pdf"):
        loader = PyMuPDFLoader(str(file_path))
        raw_docs = loader.load()

        for doc in raw_docs:
            doc.metadata["source"] = file_path.name

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100,
            separators=["\n\n", "\n", ".", " ", ""],
        )
        chunks = splitter.split_documents(raw_docs)
        all_docs.extend(chunks)

    store = LocalFileStore(f"./.cache/embeddings/all_static/schoolviolence")
    embeddings = OpenAIEmbeddings()
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, store)
    vectorstore = FAISS.from_documents(all_docs, cached_embeddings)

    return vectorstore  # as_retriever 대신 직접 vectorstore 반환

# ✅ 메시지 저장/전송
def save_message(message, role):
    st.session_state["messages1"].append({"message": message, "role": role})

def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)

def paint_history():
    for message in st.session_state["messages1"]:
        send_message(message["message"], message["role"], save=False)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# ✅ 프롬프트 템플릿
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a supervisor at the Seoul Metropolitan Office of Education, responsible for school violence prevention and student life guidance.
Answer questions based strictly on the provided context.
Your answers must follow these guidelines:
- Provide accurate, detailed, and rich information
- Use clear and professional language
- If the information is not available in the context, respond with: "문서에서 찾을 수 없습니다."
DON'T make anything up.\n\nContext: {context}"""),
    ("human", "{question}"),
])

# ✅ 페이지 UI 설정
st.set_page_config(page_title="학교폭력사안 챗봇", page_icon="📂")
st.title("📂 학교폭력사안 챗봇")
st.markdown("##### 각종 학교폭력사안에 대해 문서(**사안처리가이드북**)를 근거로 대답합니다.")

if "messages1" not in st.session_state:
    st.session_state["messages1"] = []

# ✅ 문서 로딩
vectorstore = load_all_documents()
send_message("모든 문서가 로드되었습니다. 질문하세요.", "ai", save=False)
paint_history()

# ✅ 사용자 입력 처리
message = st.chat_input("Ask about the documents...")
if message:
    send_message(message, "human")

    docs_with_scores = vectorstore.similarity_search_with_score(message, k=10)
    filtered_docs = [doc for doc, score in sorted(docs_with_scores, key=lambda x: x[1]) if score <= 0.6]
    top_docs = filtered_docs[:4]

    chain = (
        {
            "context": lambda _: format_docs(top_docs),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
    )

    with st.chat_message("ai"):
        chain.invoke(message)

    with st.expander("🔍 참고문헌", expanded=False):
        for i, doc in enumerate(top_docs, start=1):
            source = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", 0)
            st.markdown(f"**문서 {i} — 📄 `{source}` | Page {page + 1} | 유사도 점수 기준 상위**")
            st.code(doc.page_content.strip(), language="markdown")

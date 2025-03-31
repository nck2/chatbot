import os
from pathlib import Path
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.embeddings import CacheBackedEmbeddings
# from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI


from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.storage import LocalFileStore
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
# from langchain_community.chat_models import ChatOpenAI
from langchain.callbacks.base import BaseCallbackHandler
import streamlit as st



# def get_api_key():
#     # st.secrets.get은 None 반환하므로 안전
#     cloud_key = st.secrets["OPENAI_API_KEY"]
#     if cloud_key:
#         return cloud_key

#     # 로컬용 .env 환경 변수
#     from dotenv import load_dotenv
#     load_dotenv()
#     return os.getenv("OPENAI_API_KEY")


# api_key = get_api_key()



if "logged_in" not in st.session_state or not st.session_state.logged_in:
    st.error("🚫 로그인해야 사용가능합니다. Home 으로 가세요.")
    # st.markdown("[Back to Login](./)")
    st.markdown("""
    <a href="./" target="_self">🔙 Home으로 가서 로그인하기</a>
""", unsafe_allow_html=True)

    st.stop()

# ✅ 폴더 안의 모든 PDF 문서를 대상
STATIC_DIR = Path("./static/schoolviolence")

class ChatCallbackHandler(BaseCallbackHandler):
    message = ""
    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()
    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai")
    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)


llm = ChatOpenAI(
    temperature=0.1,
    streaming=True,
    callbacks=[ChatCallbackHandler()],

)

# ✅ 여러 문서 로딩 & 벡터 생성
@st.cache_resource(show_spinner="문서 로딩중. 잠시만 기다리세요...")
def load_all_documents():
    all_docs = []
    for file_path in STATIC_DIR.glob("*.pdf"):
        loader = PyMuPDFLoader(str(file_path))
        raw_docs = loader.load()

        # 각 문서에 file_name 추가
        for doc in raw_docs:
            doc.metadata["source"] = file_path.name

        splitter = CharacterTextSplitter.from_tiktoken_encoder(
            separator="\n", chunk_size=600, chunk_overlap=100,
        )
        chunks = splitter.split_documents(raw_docs)
        all_docs.extend(chunks)

    # 벡터화
    store = LocalFileStore(f"./.cache/embeddings/all_static/schoolviolence")
    embeddings = OpenAIEmbeddings()
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, store)
    vectorstore = FAISS.from_documents(all_docs, cached_embeddings)
    print("📄 문서 임베딩 실행됨")
    return vectorstore.as_retriever()

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

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Answer the question using ONLY the following context. If the answer is not found in the context below, reply in Korean with: '문서에서 찾을 수 없습니다.' The information cannot be found in the document. DON'T make anything up.\n\nContext: {context}"),
        ("human", "{question}"),
    ]
)

# ✅ Streamlit 시작
st.set_page_config(page_title="학교폭력사안 챗봇", page_icon="📂")
st.title("📂 학교폭력사안 챗봇")
st.markdown("##### 각종 학교폭력사안에 대해 문서(**사안처리가이드북, 알쓸소중**)를 근거로 대답합니다.")

# 세션 초기화
if "messages1" not in st.session_state:
    st.session_state["messages1"] = []



# 문서 불러오기
retriever = load_all_documents()
send_message("모든 문서가 로드되었습니다. 질문하세요.", "ai", save=False)
paint_history()

message = st.chat_input("Ask about the documents...")
if message:
    send_message(message, "human")
    # relevant_docs = retriever.get_relevant_documents(message)
    relevant_docs = retriever.invoke(message)


    chain = (
        {
            "context": lambda _: format_docs(relevant_docs),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
    )

    with st.chat_message("ai"):
        chain.invoke(message)

    # ✅ 각 문서 + 페이지 번호 표시
    with st.expander("🔍 참고문헌", expanded=False):
        for i, doc in enumerate(relevant_docs, start=1):
            source = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", "❓")
            st.markdown(f"**문서 {i} — 📄 `{source}` | Page {page+1}**")
            st.code(doc.page_content.strip(), language="markdown")

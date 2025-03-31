import streamlit as st
import time

# 세션 상태 초기화
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username = None

# 로그인 페이지
def login():
    st.title("🔐 Login Page")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    db_user = st.secrets["user"]
    db_pass = st.secrets["password"]

    if st.button("Login"):
        if username == db_user and password==db_pass:
            st.session_state.logged_in = True
            st.session_state.username = username
            st.success("✅ Login successful!")
            # st.stop()
            st.rerun()
        else:
            st.error("❌ Invalid credentials")

# 로그인 여부에 따라 화면 분기
if st.session_state.logged_in:
    st.success(f"Welcome, {st.session_state.username}!")
    st.markdown("옆의 사이드바에서 원하는 페이지로 가세요.!")


    if st.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.username = None
        st.experimental_rerun()
else:
    login()

import streamlit as st

# 세션 상태 초기화
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "password_input" not in st.session_state:
    st.session_state.password_input = ""
if "trigger_rerun" not in st.session_state:
    st.session_state.trigger_rerun = False

# 콜백 함수: 로그인 조건 확인만 수행
def login_callback():
    db_pass = st.secrets["password"]
    input_pass = st.session_state.password_input

    if input_pass and input_pass == db_pass:
        st.session_state.logged_in = True
        st.session_state.trigger_rerun = True
    else:
        st.error("❌ 잘못된 비번입니다.")

# 로그인 페이지
def login_page():
    st.title("🔐 Login Page")

    st.text_input(
        "Password",
        type="password",
        key="password_input",
        on_change=login_callback
    )

    if st.button("Login"):
        login_callback()

# 메인 페이지
def main_page():
    st.success("Welcome!")
    st.markdown("옆의 사이드바에서 원하는 페이지로 가세요!")

    if st.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.password_input = ""
        st.rerun()

# rerun 트리거 감지 후 실행 (콜백 바깥)
if st.session_state.get("trigger_rerun", False):
    st.session_state.trigger_rerun = False
    st.rerun()

# 페이지 분기
if st.session_state.logged_in:
    main_page()
else:
    login_page()

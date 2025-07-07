import streamlit as st

from mylog import logger, setup_logging

# Setup logging hanya sekali di aplikasi utama
if "log_config" not in st.session_state:
    setup_logging(
        log_level="INFO",
        diagnose=False,  # False untuk production
        serialize=False,  # Human-readable logs untuk development
        log_path="logs",
    )
    st.session_state.log_config = True
    logger.info("Application started - Streamlit Voucher System")

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False


def login():
    """A simple login page."""
    if st.button("Log in"):
        st.session_state.logged_in = True
        logger.info("User logged in successfully")
        st.rerun()


def logout():
    """A simple logout page."""
    if st.button("Log out"):
        st.session_state.logged_in = False
        logger.info("User logged out")
        st.rerun()


login_page = st.Page(login, title="Log in", icon=":material/login:")
logout_page = st.Page(logout, title="Log out", icon=":material/logout:")

landing = st.Page(
    "pages/landing.py", title="Landing Page", icon=":material/home:", default=True
)


ocr_page = st.Page(
    page="pages/ocr/ui_live_ocr.py", title="OCR", icon=":material/search:"
)


if st.session_state.logged_in:
    pg = st.navigation(
        {
            "Account": [logout_page],
            "Landing": [landing],
            "Tools": [ocr_page],
        }
    )
else:
    pg = st.navigation([login_page])

pg.run()

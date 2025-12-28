import streamlit as st
from dotenv import load_dotenv
import requests
from requests.auth import HTTPBasicAuth
import os

# -------------------------
# Config
# -------------------------
load_dotenv()
API_URL = os.getenv("API_URL")

st.set_page_config(
    page_title="Problem Statement - 4 ",
    layout="centered"
)

# -------------------------
# Session State
# -------------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username = ""
    st.session_state.password = ""
    st.session_state.page = "home"

# -------------------------
# Helpers
# -------------------------
def auth():
    return HTTPBasicAuth(
        st.session_state.username,
        st.session_state.password
    )

def safe_json(res):
    try:
        return res.json()
    except:
        return None

# -------------------------
# Authentication UI
# -------------------------
def auth_ui():
    st.title("📄 Enterprise PDFs → Searchable Knowledge")
    st.caption("Login to upload documents and ask questions")

    tab1, tab2 = st.tabs(["🔐 Login", "🆕 Signup"])

    # ---- LOGIN ----
    with tab1:
        st.subheader("Login")
        u = st.text_input("Username", key="login_user")
        p = st.text_input("Password", type="password", key="login_pass")

        if st.button("Login"):
            res = requests.get(
                f"{API_URL}/auth/login",
                auth=HTTPBasicAuth(u, p)
            )
            if res.status_code == 200:
                st.session_state.logged_in = True
                st.session_state.username = u
                st.session_state.password = p
                st.rerun()
            else:
                st.error("Invalid username or password")

    # ---- SIGNUP ----
    with tab2:
        st.subheader("Create Account")
        u = st.text_input("Choose Username", key="signup_user")
        p = st.text_input("Choose Password", type="password", key="signup_pass")

        if st.button("Signup"):
            res = requests.post(
                f"{API_URL}/auth/signup",
                json={"username": u, "password": p}
            )
            if res.status_code == 200:
                st.success("Account created successfully. Please login.")
            else:
                st.error("User already exists or invalid input")

# -------------------------
# Upload Page
# -------------------------
def upload_page():
    st.subheader("📤 Upload PDF Document")
    st.caption("Upload enterprise reports, manuals, or scanned PDFs")

    file = st.file_uploader("Select a PDF file", type=["pdf"])

    if st.button("Upload Document"):
        if not file:
            st.warning("Please select a PDF file")
            return

        res = requests.post(
            f"{API_URL}/docs/upload_docs",
            files={"file": (file.name, file.getvalue(), "application/pdf")},
            auth=auth()
        )

        if res.status_code == 200:
            st.success("Document uploaded and indexed successfully")
        else:
            st.error("Document upload failed")

# -------------------------
# Query Page
# -------------------------
def query_page():
    st.subheader("💬 Ask a Question")
    st.caption("Ask questions directly from uploaded documents")

    q = st.text_input("Enter your question")

    if st.button("Get Answer"):
        if not q.strip():
            st.warning("Please enter a question")
            return

        res = requests.post(
            f"{API_URL}/chat",
            data={"message": q},
            auth=auth()
        )

        data = safe_json(res)

        if res.status_code == 200:
            st.success(data.get("answer", ""))
        else:
            st.error("Failed to retrieve answer")

# -------------------------
# Extract / Summary Page
# -------------------------
def query_extract_summury():
    st.subheader("🧾 Document Summary")
    st.caption("Automatically summarize uploaded documents")

    if st.button("Extract Summary"):
        res = requests.post(
            f"{API_URL}/chat",
            data={"message": "Summarize the document"},
            auth=auth()
        )

        data = safe_json(res)

        if res.status_code == 200:
            st.success(data.get("answer", ""))
        else:
            st.error("Failed to extract summary")

# -------------------------
# Main App
# -------------------------
if not st.session_state.logged_in:
    auth_ui()
else:
    st.title(f"Welcome, {st.session_state.username}")
    st.caption("Choose an action below")

    col1, col2, col3 = st.columns(3)

    if col1.button("📤 Upload Document"):
        st.session_state.page = "upload"
    if col2.button("💬 Ask Query"):
        st.session_state.page = "query"
    if col3.button("🧾 Extract Summary"):
        st.session_state.page = "extract"

    st.divider()

    if st.session_state.page == "upload":
        upload_page()
    elif st.session_state.page == "query":
        query_page()
    elif st.session_state.page == "extract":
        query_extract_summury()

    if st.button("Logout"):
        st.session_state.clear()
        st.rerun()

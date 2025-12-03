# stream-lit_app.py
# Streamlit UI for NIVA symptom chatbot
# - Auto-finish after EXACTLY 5 patient replies
# - Generates JSON + PDF once
# - After finish: hide input, show PDF download
# - Always show Reset Conversation button

import os
import streamlit as st
import traceback
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="NIVA — Symptom Chatbot", layout="wide")
st.title("NIVA — Symptom Analysis Chatbot")


# --------- Load pipeline functions ----------
@st.cache_resource
def load_pipeline():
    try:
        import langchain_pipeline as lp

        funcs = {
            "generate_reply": getattr(lp, "generate_conversational_reply", None),
            "extract_structured": getattr(
                lp, "extract_structured_from_conversation", None
            ),
            "triage": getattr(lp, "triage_report", None),
            "save_report": getattr(lp, "save_report", None),
        }

        # PDF generator (mandatory)
        from langchain_pipeline import save_report_pdf

        funcs["save_report_pdf"] = save_report_pdf

        missing = [k for k, v in funcs.items() if v is None]
        if missing:
            raise ImportError(f"Missing required functions: {missing}")

        return funcs

    except Exception as e:
        raise ImportError(traceback.format_exc()) from e


# --------- Initialize Session State ----------
STARTER_Q = (
    "What problem are you facing? Please describe your main symptom in one sentence."
)

if "messages" not in st.session_state:
    st.session_state.messages = [("bot", STARTER_Q)]
if "finished" not in st.session_state:
    st.session_state.finished = False
if "structured" not in st.session_state:
    st.session_state.structured = None
if "json_path" not in st.session_state:
    st.session_state.json_path = None
if "pdf_path" not in st.session_state:
    st.session_state.pdf_path = None

pipeline = load_pipeline()

# --------- Layout ----------
col_main, col_side = st.columns([3, 1])

with col_main:
    st.subheader("Chat")

    # ------------------ Chat + Finish (only if NOT finished) ------------------
    if not st.session_state.finished:
        # User Input
        with st.form("chat_form", clear_on_submit=True):
            user_input = st.text_input("Your reply", placeholder="Type your reply...")
            submitted = st.form_submit_button("Send")

        # Manual Finish Button
        finish_manual = st.button("Finish Now & Generate Report")

        # Handle User Reply
        if submitted:
            if user_input.strip() == "":
                st.warning("Please type something.")
            else:
                st.session_state.messages.append(("patient", user_input.strip()))

                # Generate bot reply
                bot_reply = pipeline["generate_reply"](
                    st.session_state.messages,
                    user_input.strip(),
                    k_context=3,
                    max_turns=6,
                    temperature=0.0,
                )
                st.session_state.messages.append(("bot", bot_reply))

                # Count patient messages
                patient_count = sum(
                    1 for r, _ in st.session_state.messages if r == "patient"
                )

                # Auto Finish After 5 Patient Replies
                if patient_count >= 5:
                    st.session_state.finished = True

        # Manual Finish
        if finish_manual:
            st.session_state.finished = True

    # ------------------ Generate Report ONCE when finished ------------------
    if st.session_state.finished and st.session_state.structured is None:
        conv_text = "\n".join(
            f"{'Patient' if r == 'patient' else 'Bot'}: {t}"
            for r, t in st.session_state.messages
        )

        # Extract structured info
        structured = pipeline["extract_structured"](conv_text)
        st.session_state.structured = structured

        # Triage + Save JSON
        triage = pipeline["triage"](structured)
        out = {
            "conversation": st.session_state.messages,
            "structured": structured,
            "triage": triage,
        }
        json_path = pipeline["save_report"](out)
        st.session_state.json_path = json_path

        # Generate PDF
        pdf_path = pipeline["save_report_pdf"](out)
        st.session_state.pdf_path = pdf_path

        st.success(f"Report generated — JSON: {json_path}, PDF: {pdf_path}")

    # ------------------ Finished UI: show download instead of input ------------------
    if st.session_state.finished:
        st.markdown("### Conversation Complete")
        st.write("NIVA has collected enough information and generated your report.")

        if st.session_state.pdf_path and os.path.exists(st.session_state.pdf_path):
            with open(st.session_state.pdf_path, "rb") as f:
                pdf_bytes = f.read()
            st.download_button(
                label="Download PDF Report",
                data=pdf_bytes,
                file_name=os.path.basename(st.session_state.pdf_path),
                mime="application/pdf",
            )
        else:
            st.warning("PDF report not found on disk.")

    # ------------------ Render Chat ------------------
    for role, msg in st.session_state.messages:
        if role == "patient":
            st.markdown(f"**You:** {msg}")
        else:
            st.markdown(f"**NIVA:** {msg}")

with col_side:
    st.subheader("Status")
    st.success("Pipeline Loaded")

    # --------- RESET BUTTON (always visible) ----------
    if st.button("Reset Conversation"):
        st.session_state.messages = [("bot", STARTER_Q)]
        st.session_state.finished = False
        st.session_state.structured = None
        st.session_state.json_path = None
        st.session_state.pdf_path = None
        st.rerun()  # reload app with fresh state

    st.markdown("---")
    st.subheader("Report")
    if st.session_state.finished and st.session_state.structured:
        st.json(st.session_state.structured)
    else:
        st.info("Report will appear here after the conversation is finished.")

st.markdown("---")
st.caption("Demo: conversational CSV-guided + LLM intake (not a medical device).")

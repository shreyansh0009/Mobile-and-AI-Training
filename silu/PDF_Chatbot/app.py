import streamlit as st
import os
from backend import modelFunction, predict


st.set_page_config(page_title="PDF Chatbot", layout="centered")

# SESSION STATE
if "messages" not in st.session_state:
    st.session_state.messages = []

if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

if "pdf_path" not in st.session_state:
    st.session_state.pdf_path = None

# CHAT BUBBLE 
def chat_bubble(message, sender="bot"):
    if sender == "user":
        bg = "#DCF8C6"
        align = "flex-end"
        label = "You"
    else:
        bg = "#F1F0F0"
        align = "flex-start"
        label = "Bot"

    st.markdown(
        f"""
        <div style="display:flex; justify-content:{align}; margin:6px 0;">
            <div style="
                background:{bg};
                padding:10px 12px;
                border-radius:10px;
                max-width:70%;
                font-size:14px;
                line-height:1.4;
            ">
                <b>{label}:</b><br>{message}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

# HEADER 
st.markdown("<h2 style='text-align:center;'>🤖 PDF Chatbot</h2>", unsafe_allow_html=True)

# # PDF UPLOADER 
uploaded_pdf = st.file_uploader(
    "Upload any PDF",
    type=["pdf"]
)

def save_uploaded_pdf(uploaded_file):
    os.makedirs("uploads", exist_ok=True)
    path = os.path.join("uploads", uploaded_file.name)
    with open(path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return os.path.abspath(path)

# # PROCESS PDF 

if uploaded_pdf and st.session_state.pdf_path is None:
    st.session_state.pdf_path = save_uploaded_pdf(uploaded_pdf)
    with st.spinner("Indexing PDF..."):
        st.session_state.rag_chain = modelFunction(st.session_state.pdf_path)
    st.success("PDF ready for chat ")



# uploaded_pdf = st.file_uploader("Upload PDF", type=["pdf"])

# if uploaded_pdf and st.session_state.rag_chain is None:
#     with st.spinner("Indexing PDF in memory..."):
#         st.session_state.rag_chain = modelFunction(uploaded_pdf.getvalue())
#     st.success("PDF ready (in-memory) ")

# # CHAT DISPLAY 
chat_container = st.container()

with chat_container:
    for msg in st.session_state.messages:
        chat_bubble(msg["content"], msg["role"])

# # INPUT AREA
question = st.text_input("Ask any question related to the PDF")

if st.button("Send"):
    if not question.strip():
        st.warning("Please enter a question")
    elif not st.session_state.rag_chain:
        st.warning("Please upload a PDF first")
    else:
        st.session_state.messages.append(
            {"role": "user", "content": question}
        )

        with st.spinner("Thinking..."):
            answer = predict(st.session_state.rag_chain, question)
            question = ''
        st.session_state.messages.append(
            {"role": "bot", "content": answer}
        )

        st.rerun()

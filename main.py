import os
import streamlit as st
from utils.transcriber import transcribe_video
from utils.rag import create_rag

st.set_page_config(
    page_title="AI Video Assistant",
    page_icon="🎙️",
    layout="wide"
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Developed by Nihal Tiwari**")
st.sidebar.markdown("© 2026 All Rights Reserved")



UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

st.title("🎙️ AI Meeting Assistant")

uploaded_file = st.file_uploader(
    "Upload Meeting Recording",
    type=["mp4", "mov", "mkv"]
)

if uploaded_file is not None:

    save_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)

    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success("Video uploaded successfully!")
    st.video(save_path)

    # Process only once

    with st.spinner("Processing meeting..."):

            # Transcribe (don't display transcript)
            _, transcript_path = transcribe_video(save_path)

            # Build RAG
            st.session_state.rag_chain = create_rag(transcript_path)

    st.success("Meeting processed successfully!")

    question = st.text_input("Ask a question about the meeting")

    if question:
        answer = st.session_state.rag_chain.invoke(question)

        st.subheader("Answer")
        st.write(answer)


# ---------------- Footer ----------------
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray; padding: 10px;'>
        © 2026 Nihal Tiwari | AI Meeting Assistant
    </div>
    """,
    unsafe_allow_html=True,
)

# 🎙️ AI Meeting Assistant

An AI-powered Meeting Assistant that transcribes meeting recordings, builds a Retrieval-Augmented Generation (RAG) knowledge base, and allows users to ask questions about the meeting using a Large Language Model (LLM).

---

## 🚀 Features

- 📹 Upload meeting recordings (`.mp4`, `.mov`, `.mkv`)
- 🎤 Automatic speech-to-text transcription using OpenAI Whisper
- 📄 Converts meeting recordings into searchable transcripts
- 🧠 Builds a RAG pipeline using LangChain and ChromaDB
- 🤖 Answers meeting-related questions using Groq Llama 3.1
- ⚡ Fast semantic search with HuggingFace sentence embeddings
- 🌐 Simple and interactive Streamlit web interface

---

## 🛠️ Tech Stack

- **Frontend:** Streamlit
- **Speech Recognition:** OpenAI Whisper
- **LLM:** Groq (Llama 3.1 8B Instant)
- **RAG Framework:** LangChain
- **Vector Database:** ChromaDB
- **Embeddings:** sentence-transformers (all-MiniLM-L6-v2)
- **Language:** Python

---

## 📂 Project Structure

```
Video_Summarization/
│
├── main.py
├── requirements.txt
├── packages.txt
├── README.md
│
├── utils/
│   ├── rag.py
│   └── transcriber.py
│
├── uploads/
├── transcripts/
```

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/Video_Summarization.git

cd Video_Summarization
```

### 2. Create a Virtual Environment

Windows

```bash
python -m venv .venv

.venv\Scripts\activate
```

Linux / macOS

```bash
python3 -m venv .venv

source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🔑 Environment Variables

Create a `.env` file in the project root.

```env
GROQ_API_KEY=your_groq_api_key
```

Get your API key from:

https://console.groq.com/keys

---

## ▶️ Run the Application

```bash
streamlit run main.py
```

The application will open in your browser.

---

## 📌 How It Works

1. Upload a meeting recording.
2. Whisper transcribes the video into text.
3. The transcript is split into chunks.
4. HuggingFace embeddings convert chunks into vectors.
5. ChromaDB stores the embeddings.
6. LangChain retrieves relevant transcript chunks.
7. Groq Llama generates answers based on the retrieved context.

---

## 🧠 RAG Pipeline

```
Meeting Video
      │
      ▼
OpenAI Whisper
      │
      ▼
Meeting Transcript
      │
      ▼
Text Splitter
      │
      ▼
SentenceTransformer Embeddings
      │
      ▼
ChromaDB
      │
      ▼
Retriever
      │
      ▼
Groq Llama 3.1
      │
      ▼
Meeting Question Answering
```

---

## 📷 Screenshots

### Home Page

> Add a screenshot here.

### Meeting Upload

> Add a screenshot here.

### Question Answering

> Add a screenshot here.

---

## 📦 Deployment

The application can be deployed on **Streamlit Community Cloud**.

### `packages.txt`

```
ffmpeg
```

### Add Secret

```
GROQ_API_KEY=your_groq_api_key
```

Deploy directly from your GitHub repository.

---

## 🔮 Future Improvements

- Meeting summarization
- Speaker diarization
- Timestamped transcripts
- PDF/DOCX export
- Multi-language transcription
- Conversation memory
- Chat history
- Support for YouTube videos
- Meeting action item extraction

---

## 👨‍💻 Author

**Nihal Tiwari**

AI & Machine Learning Developer

GitHub: https://github.com/your-github-username

LinkedIn: https://linkedin.com/in/your-linkedin-profile

---

## 📄 License

This project is licensed under the MIT License.

---

⭐ If you found this project helpful, consider giving it a star on GitHub!
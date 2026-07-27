import os

from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()


def create_rag(transcript_path):

    # Load transcript
    loader = TextLoader(transcript_path, encoding="utf-8")
    documents = loader.load()

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(documents)

    # Embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    from langchain_chroma import Chroma

    # Delete previous collection (if it exists)
    try:
        Chroma(
            collection_name="meeting",
            embedding_function=embeddings
        ).delete_collection()
    except:
        pass

    # Create a fresh collection
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="meeting"
    )

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 4,
            "fetch_k": 8
        }
    )
    # LLM
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0
    )

    prompt = ChatPromptTemplate.from_template("""
You are an AI Meeting Assistant.

Use ONLY the information provided in the transcript.

If the answer is not present in the transcript, reply:
"I couldn't find that information in the meeting transcript."

Transcript:
{context}

Question:
{input}

Answer:
""")

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
            {
                "context": retriever | format_docs,
                "input": RunnablePassthrough(),
            }
            | prompt
            | llm
            | StrOutputParser()
    )

    return rag_chain

import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from langchain_core.runnables import RunnableLambda
from pypdf import PdfReader
from langchain_core.documents import Document
from io import BytesIO


load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv('INDEX_NAME')
PINECONE_ENVIRONMENT = os.getenv('PINECONE_ENVIRONMENT')

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found")

if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY not found")

#PINECONE 
def init_pinecone():
    pc = Pinecone(api_key=PINECONE_API_KEY)

    if INDEX_NAME not in pc.list_indexes().names():
        pc.create_index(
            name=INDEX_NAME,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region=PINECONE_ENVIRONMENT
            )
        )
    return pc


# MODELS
def create_embeddings():
    return OpenAIEmbeddings(
        model="text-embedding-3-small"
    )

def create_llm():
    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0
    )

#  PDF INGEST
def ingest_pdf(pdf_path, embeddings):
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(docs)

    vectorstore = PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=INDEX_NAME
    )
    return vectorstore

# PROMPT 
def create_prompt():
    return PromptTemplate.from_template(
         """
        You are a helpful assistant answering questions about a PDF document.

        Use ONLY the context below to answer.
        - If the question is a greeting, respond politely.
        - If the user asks to describe or summarize the document, give a brief summary based on the context.
        - If the answer truly does not exist in the context, say:
          "I cannot find this information in the provided document."

        Context:
        {context}

        Question:
        {question}
        """
    )


def format_docs(docs):
    return "\n".join(doc.page_content for doc in docs)

#RAG 
def build_rag_chain(vectorstore, llm, prompt):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    return (
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
    )


# PUBLIC FUNCTIONS 
def modelFunction(pdf_path):
    init_pinecone()

    embeddings = create_embeddings()
    llm = create_llm()
    prompt = create_prompt()

    vectorstore = ingest_pdf(pdf_path, embeddings)
    rag_chain = build_rag_chain(vectorstore, llm, prompt)

    return rag_chain


# def ingest_pdf_from_bytes(pdf_bytes, embeddings):
#     reader = PdfReader(pdf_bytes)

#     documents = []
#     for i, page in enumerate(reader.pages):
#         text = page.extract_text()
#         if text:
#             documents.append(
#                 Document(
#                     page_content=text,
#                     metadata={"page": i + 1}
#                 )
#             )

#     splitter = RecursiveCharacterTextSplitter(
#         chunk_size=1000,
#         chunk_overlap=200
#     )
#     chunks = splitter.split_documents(documents)

#     vectorstore = PineconeVectorStore.from_documents(
#         documents=chunks,
#         embedding=embeddings,
#         index_name=INDEX_NAME
#     )
#     return vectorstore


# def modelFunction(pdf_bytes):
#     init_pinecone()

#     embeddings = create_embeddings()
#     llm = create_llm()
#     prompt = create_prompt()

#     vectorstore = ingest_pdf_from_bytes(BytesIO(pdf_bytes), embeddings)
#     rag_chain = build_rag_chain(vectorstore, llm, prompt)

#     return rag_chain

def predict(rag_chain, question):
    response = rag_chain.invoke(question)
    return response.content

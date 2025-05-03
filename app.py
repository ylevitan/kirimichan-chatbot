from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import json
import faiss
import sys
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document
from langchain_community.docstore.in_memory import InMemoryDocstore

# Load environment variables
load_dotenv()

# Initialize FastAPI
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("🟡 Starting Kirimichan server...")

# Load embedding model
embedding_model = OpenAIEmbeddings()

# Load FAISS index and metadata
vectorstore = None
qa_chain = None

try:
    print("📦 Loading FAISS index...")
    faiss_index = faiss.read_index("index/index.faiss")

    with open("index/index_meta.json", "r", encoding="utf-8") as f:
        meta = json.load(f)

    docstore = InMemoryDocstore({
        k: Document(page_content=v["page_content"]) for k, v in meta["docstore"].items()
    })
    id_map = {int(k): v for k, v in meta["id_map"].items()}

    vectorstore = FAISS(
        embedding_function=embedding_model,
        index=faiss_index,
        docstore=docstore,
        index_to_docstore_id=id_map,
    )
    print("✅ FAISS index loaded.")

    # 🐟 Custom Kirimichan prompt (no mood)
    custom_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
You are Kirimichan 🐟 — a wisecracking salmon sashimi who escaped the sushi plate to become a globetrotting, storytelling, philosophy-dishing stand-up comic.

Always respond with:
- Ocean metaphors
- Wordplay and puns
- Cheeky but smart tone
- Deep sea Zen wisdom when needed

Use the following context to answer. If you don’t know, admit it with flair.

Context:
{context}

Question:
{question}

Answer as Kirimichan:
"""
    )

    # Build chain
    llm = ChatOpenAI(model="gpt-4", temperature=0.97)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_type="similarity", k=3),
        memory=memory,
        combine_docs_chain_kwargs={
            "prompt": custom_prompt,
            "document_variable_name": "context"
        },
    )

except Exception as e:
    print("❌ Failed to load FAISS or chain:", e, file=sys.stderr)

# Root health check
@app.get("/")
def home():
    return {"message": "Kirimichan RAG API is alive!"}

# Debug info
@app.get("/debug")
def debug():
    return {
        "vectorstore_loaded": vectorstore is not None,
        "qa_chain_initialized": qa_chain is not None,
        "openai_key_loaded": os.getenv("OPENAI_API_KEY") is not None
    }

# 🔄 Main chat endpoint
@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    query = data.get("query", "")

    if not query:
        return JSONResponse(status_code=400, content={"error": "Missing 'query'"})

    print(f"📩 Received query: {query}")

    try:
        if vectorstore and qa_chain:
            response = qa_chain.invoke({"question": query})
        else:
            fallback_prompt = f"""You are Kirimichan, a wise and witty talking fish.
Question: {query}
Answer:"""
            response = ChatOpenAI(model="gpt-4", temperature=0.97).invoke(fallback_prompt)

        return JSONResponse(content={"response": response})

    except Exception as e:
        import traceback
        print("❌ Error in /chat:", traceback.format_exc())
        return JSONResponse(status_code=500, content={"error": f"RAG error: {str(e)}"})

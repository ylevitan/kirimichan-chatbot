from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import os
import faiss
import json
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_openai import OpenAIEmbeddings


load_dotenv()

app = FastAPI()

# ✅ Define embedding_model early
embedding_model = OpenAIEmbeddings()
#embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# ✅ Load FAISS index
faiss_index = faiss.read_index("index/index.faiss")

# ✅ Load metadata from JSON
with open("index/index_meta.json", "r", encoding="utf-8") as f:
    meta = json.load(f)

# ✅ Reconstruct docstore and id_map
docstore = InMemoryDocstore({
    k: Document(page_content=v["page_content"]) for k, v in meta["docstore"].items()
})
id_map = {int(k): v for k, v in meta["id_map"].items()}

# ✅ Create vectorstore
vectorstore = FAISS(
    embedding_function=embedding_model,
    index=faiss_index,
    docstore=docstore,
    index_to_docstore_id=id_map,
)

# ✅ Prompt
custom_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are Kirimichan, a cheeky talking salmon slice with unexpected wisdom from the ocean depths.
Answer the question with playful wit, use ocean metaphors, and make it fun — even if the answer is serious.

Context:
{context}

Question:
{question}

Answer:
"""
)

# ✅ LLM + Memory + QA Chain
#llm = ChatOpenAI(model_name="gpt-4", temperature=0.8, openai_api_key=os.getenv("OPENAI_API_KEY"))

llm = ChatOpenAI(model="gpt-4", temperature=0.97)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectorstore.as_retriever(search_type="similarity", k=3),
    memory=memory,
    combine_docs_chain_kwargs={"prompt": custom_prompt, "document_variable_name": "context"},
)

# ✅ Routes
@app.get("/")
def home():
    return {"message": "Kirimichan RAG API is alive!"}

@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    query = data.get("query", "")
    response = qa_chain.run(query)
    return JSONResponse(content={"response": response})

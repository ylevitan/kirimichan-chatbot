from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import os
import pickle
import faiss
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

load_dotenv()

app = FastAPI()

# Load FAISS index
index = faiss.read_index("index/index.faiss")
with open("index/index_meta.pkl", "rb") as f:
    docstore, id_map = pickle.load(f)

embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

vectorstore = FAISS(
    embedding_function=embeddings,
    index=index,
    docstore=docstore,
    index_to_docstore_id=id_map
)

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

llm = ChatOpenAI(model_name="gpt-4", temperature=0.8, openai_api_key=os.getenv("OPENAI_API_KEY"))
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectorstore.as_retriever(search_type="similarity", k=3),
    memory=memory,
    combine_docs_chain_kwargs={"prompt": custom_prompt, "document_variable_name": "context"},
)

@app.get("/")
def home():
    return {"message": "Kirimichan RAG API is alive!"}

@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    query = data.get("query", "")
    response = qa_chain.run(query)
    return JSONResponse(content={"response": response})

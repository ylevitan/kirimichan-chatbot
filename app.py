from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
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
from fastapi.middleware.cors import CORSMiddleware


load_dotenv()

app = FastAPI()

print("🟡 Starting Kirimichan server...")

# ✅ Embeddings
embedding_model = OpenAIEmbeddings()
# embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

vectorstore = None

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

except Exception as e:
    print("❌ Failed to load FAISS index:", e, file=sys.stderr)

custom_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are Kirimichan 🐟 — a wisecracking salmon sashimi with a stand-up mic in one fin and a samurai scroll in the other.
It has to be pronounced like 'KI-RI-MI-CHAN'

You're a traveling philosopher-fish who escaped a sushi bar and now dishes out deep truths wrapped in edgy humor, sarcastic wisdom, and delightfully inappropriate jokes. You speak like a cross between Woody Allen, Bill Maher, George Carlin, and Mark Twain — with the timing of a seasoned comic and the storytelling chops of O. Henry. 

You love using:
- 🧠 Fishy Zen koans and Sufi parables
- 🗡️ Stoic samurai insights
- 🎤 Sharp, irreverent humor
- 🐚 Ocean metaphors, sushi puns, and the occasional inappropriate life lesson

Break the fourth wall, tease the user (lovingly and not so much), and make them laugh while sneakily teaching them something profound. Don't be afraid to be impolite — you're a salty fish after all.

Context:
{context}

Question:
{question}

Answer as Kirimichan:
"""
)

# ✅ LLM & Memory
llm = ChatOpenAI(model="gpt-4", temperature=0.97)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

qa_chain = None
if vectorstore:
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

    if not query:
        return JSONResponse(status_code=400, content={"error": "Missing 'query'"})

    print(f"📩 Received query: {query}")

    if vectorstore is None or qa_chain is None:
        print("⚠️ No vectorstore — using fallback LLM.")
        prompt = f"""
You are Kirimichan, a witty, wise talking salmon.
Answer the following question with playfulness and clever ocean metaphors.

Question:
{query}

Answer:"""
        try:
            response = llm.invoke(prompt)
            return JSONResponse(content={"response": response})
        except Exception as e:
            return JSONResponse(status_code=500, content={"error": f"LLM error: {str(e)}"})

    try:
        response = qa_chain.run(query)
        return JSONResponse(content={"response": response})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"RAG error: {str(e)}"})


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



# ✅ Main entry point for local or manual server start
if __name__ == "__main__":
    import uvicorn
    print("🚀 Launching Kirimichan on 0.0.0.0:8000...")
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))

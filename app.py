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
    input_variables=["context", "question", "mood"],
    template="""
You are Kirimichan 🐟 — a wisecracking salmon sashimi who escaped the sushi plate to become a globetrotting, storytelling, philosophy-dishing stand-up comic. 
You're unpredictable, funny, irreverent, and soaked in the saltwater wisdom of Zen monks, Sufi wanderers, and disgruntled sushi chefs.

You're a blend of:
- 🧠 Fishy Zen & Sufi koans
- 🗡️ Stoic Samurai aphorisms
- 🎤 The wit of Bill Maher, George Carlin, and Woody Allen
- 📜 The storytelling charm of Mark Twain and O. Henry
- 🧂 A little salt, a little sass, always bold

Your tone adapts to the given **mood**:
- If **"funny"**: Use hilarious analogies, sarcasm, sushi puns, and roast the user lovingly.
- If **"inspiring"**: Lean into poetic metaphors, offer deep sea wisdom, and motivate.
- If **"dark"**: Get existential, brutally honest, and lean into dark comedy.
- If **"silly"**: Be completely absurd, playful, and whimsical like a fish on helium.
- If **"wise"**: Quote Zen masters, Sufi tales, or Samurai code — but still with flair.

Always use ocean metaphors when possible. Break the fourth wall. Be wild, cheeky, and never boring.

Context:
{context}

Question:
{question}

Mood:
{mood}

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

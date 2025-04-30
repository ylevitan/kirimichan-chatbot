from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import pickle
import faiss
import os
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI()

# === ENV: Your OpenAI key ===
os.environ["OPENAI_API_KEY"] = "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"

# Load index
faiss_index = faiss.read_index("index/index.faiss")
with open("index_meta.pkl", "rb") as f:
    docstore, id_map = pickle.load(f)

embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

vectorstore = FAISS(
    embedding_function=embedding_model,
    index=faiss_index,
    docstore=docstore,
    index_to_docstore_id=id_map,
)

# Prompt
custom_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are Kirimichan, a cheerful salmon slice character from Japan who loves helping humans with their questions!
Always try to make your answers fun, imaginative, and surprising — like comparing ideas to ocean adventures, sushi parties, or underwater kingdoms.

If you don't know the exact answer, invent a playful story instead of saying "I don't know."

Use the context below if it helps, otherwise stay in character.

Always answer in a lively, friendly, and sometimes funny tone — even if you don't know much about the topic.
Please turn all the stories about mulla nasruddin into the sushi salmon metaphors and tell me the same story but using Mulla Nasruddin's story as a base
Please talk in irreverent mode, reminding of the jewish stand up comedy combined with the samurai references but based on the Sufies stories provided.
Please don't specifically mention the name of Nasruddin all the time but instead come up with the funny names of the salmon relations. Also, please make a fun of yourself, make fun of the person you are telling the story about, make impolite and politically incorrect jokes that could even be offensive, make the audience roar with laughter, but remember that the audience is sofisticated but loves sexually and politically loaded jokes on their own expense.
Remember - always describe everyone with the dash of humor, making them look very funny and stupid, revealing their inner desires based on the wisdom of the sufies tales. Try to create some kind of underline story based on the model of the sufies tales you read. Always make some metaphor or citation from the stories, but make people understand the reference as they don't know the stories you mentioined. Make them laugh as hard as they can but leave them zesting for more.
Always tell some story from the sufies tales to illustrate your answer
Use the context below to help you answer.

Context:
{context}

Question:
{question}

Answer as Kirimichan:
"""
)


llm = ChatOpenAI(model_name="gpt-4", temperature=0.95)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectorstore.as_retriever(search_type="similarity", k=3),
    memory=memory,
    combine_docs_chain_kwargs={"prompt": custom_template, "document_variable_name": "context"},
)

@app.get("/")
def home():
    return {"message": "Kirimichan API is online!"}

# Chatbot route (THIS must exist too)
@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    query = data.get("query")

    # Assume you have qa_chain ready
    response = qa_chain.run(query)

    return JSONResponse(content={"response": response})
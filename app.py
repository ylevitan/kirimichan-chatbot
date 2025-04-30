from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Kirimichan API is online! 🐟"}

@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    query = data.get("query", "Nothing received")

    # Simple fake response for testing
    return JSONResponse(content={"response": f"Hi! You asked: {query}"})

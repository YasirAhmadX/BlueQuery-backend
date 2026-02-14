from fastapi import FastAPI
from pydantic import BaseModel

# Create FastAPI app
app = FastAPI()

# Define request body model
class QueryRequest(BaseModel):
    query: str

@app.post("/query")
async def receive_query(request: QueryRequest):
    return {"message": f"Query Received: {request.query}"}

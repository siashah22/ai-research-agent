from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from agent import run_research, vectorstore
from memory import retrieve_related
import uvicorn

app = FastAPI(
    title = "AI Research Agent",
    description = "An agentic research assistant with persistant memory",
    version = "1.0.0"
)


# REQUEST / RESPONSE MODELS
class ResearchRequest(BaseModel):
    query: str
    
class ResearchResponse(BaseModel):
    query: str
    summary: str
    used_memory: bool = False
    
class MemoryResponse(BaseModel):
    query: str
    related_research: str
    
# ROUTES
@app.get("/",include_in_schema=False)
def frontend():
    return FileResponse("static/index.html")

@app.post("/research", response_model=ResearchResponse)
def research(request: ResearchRequest):
    """
    Run the research agent on a query.
    Returns a structured summary with sources.
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    past = retrieve_related(vectorstore, request.query)
    used_memory = bool(past)
    
    summary = run_research(request.query)
    
    return ResearchResponse(
        query=request.query,
        summary=summary,
        used_memory=used_memory
    )
    
@app.get("/memory")
def get_memory(query: str):
    """
    Retrieve past research related to a query without running the agent.
    Useful for checking what the agent already knows.
    """
    if not query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    related = retrieve_related(vectorstore, query)
    return MemoryResponse(
        query=query,
        related_research=related if related else "No related research found."
    )
    
@app.get("/health")
def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
    
import os
import json
import time
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

# EMBEDDING MODEL
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

MEMORY_PATH = "memory_store"

# LOAD OR CREATE MEMORY
def load_memory():
    """Load existing memory from disk, or create a fresh one."""
    if os.path.exists(MEMORY_PATH):
        print("Loading existing memory...")
        return FAISS.load_local(
            MEMORY_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )
    print("Creating fresh memeory...")
    # FAISS needs at least one document to initialise
    initial_doc = Document(
        page_content="Memory initialised.",
        metadata={"query": "init", "timestamp": time.time()}
    )
    return FAISS.from_documents([initial_doc], embeddings)

def save_memory(vectorstore):
    """Save memory to disk so it persists across sessions."""
    vectorstore.save_local(MEMORY_PATH)
    print("Memory Saved.")
    
# SAVE A RESEARCH SESSION
def save_research(vectorstore, query:str, summary:str):
    """Store a research query + summary into memory."""
    doc = Document(
        page_content=f"Query: {query}\n\nSummary: {summary}",
        metadata={
            "query": query,
            "timestamp": time.time()
        }
    )
    vectorstore.add_documents([doc])
    save_memory(vectorstore)
    print(f"Research on '{query}' saved to memory.")
    
# RETRIEVE RELATED PAST RESEARCH
def retrieve_related(vectorstore, query: str, k: int=3, threshold: float=0.5)->str:
    """Find the relevant past research - only return if similarity is high enough"""
    docs_and_score = vectorstore.similarity_search_with_score(query, k=k)
    relevant = [
        doc for doc,score in docs_and_score
        if doc.metadata.get("query") != "init" and score < threshold]
    
    if not relevant:
        return ""
    
    print(f"Found {len(relevant)} relevant memory(s) for: '{query}'\n")
    
    return "\n\n---\n\n".join([doc.page_content for doc in relevant])

if __name__ == "__main__":
    vs=load_memory()
    
    #test case
    save_research(vs,"AI agents 2025", "AI agents are becoming mainstream in 2025...")
    
    result=retrieve_related(vs, "what do you know about AI agents ?")
    print("\nRetrieved Memory:\n")
    print(result)
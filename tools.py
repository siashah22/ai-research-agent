from langchain_tavily import TavilySearch
from langchain_core.tools import tool
from dotenv import load_dotenv 

load_dotenv()
# hidden - not exposed to the agent directly
_tavily = TavilySearch(max_results=5)

@tool 
def web_search(query:str)->str:
    """Search the web for current information on any topic.
    Use this when you need recent news, articles, or general information.
    Input must be a single search query string and nothing else.
    """
    result = _tavily.invoke({"query": query})
    if isinstance(result,dict) and "results" in result:
        outputs = []
        for r in result["results"]:
            outputs.append(f"Source: {r['url']}\n{r['content']}")
        return "\n\n".join(outputs)
    return str(result)
    
@tool
def summarize_topic(text:str)->str:
    """
    Use this when you have collected enough information and are ready to write the final summary. 
    Input should be all the raw information you have gathered so far.
    """
    return text

tools=[web_search, summarize_topic]
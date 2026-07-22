from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from dotenv import load_dotenv
from tools import tools
from memory import load_memory, save_research, retrieve_related

load_dotenv()
vectorstore = load_memory()
# THE BRAIN 
llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,
).bind_tools(tools)

# TOOL MAP - let's us call tools by name

tool_map = {tool.name: tool for tool in tools}

# THE RUNNER
def run_research(query:str)->str:
    print(f"\nResearching : {query}\n")
    
    past_research = retrieve_related(vectorstore, query)
    
    if past_research:
        memory_context = f"""You have researched related topics before.
        Here is what you found previously - use this to enrich your answer 
        but still search for new, updated information:
        
        {past_research}
        
        ---
        """
    else:
        memory_context=""
    sys_msg = SystemMessage(content="""
        You are an expert AI research assistant. 
        Your job is to research any topic thoroughly using the tools available
        and produce a clear, structured summary.
        
        {memory_context}
    
    Rules:
    - Always search at least 3 times before summarizing
    - Each search query must be different and more specific than the last
    - Never make up information - only use what you found in search results
    - Final answer must be detailed, structured, and use bullet points
    """)
    messages = [sys_msg, HumanMessage(content=query)]
    for iteration in range(10):
        response = llm.invoke(messages)
        messages.append(response)
        
        if not response.tool_calls:
            print("\nResearch Complete.\n")
            save_research(vectorstore, query, response.content)
            return response.content
        
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            tool_id = tool_call["id"]
            
            print(f"Searching: {tool_args.get('query', tool_args)}")
            
            # run the tool
            tool_result = tool_map[tool_name].invoke(tool_args)
            
            # feed the result back to the conversation
            messages.append(ToolMessage(
                content=str(tool_result),
                tool_call_id = tool_id
            ))
    return "Max Iterations reached"
if __name__ == "__main__":
    while True:
        print("\n" + "-"*40)
        query = input("What do you want to research ? (or 'quit') -> ")
        if query.lower() == "quit":
            print("GOODBYE !")
            break
    
        output = run_research(query)
        print("\n"+"-"*40)
        print("FINAL RESEARCH SUMMARY")
        print("\n"+"-"*40)
        print(output)

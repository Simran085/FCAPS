 # /RAG/agent_tools.py

from langchain.tools import Tool
from mistral_llm_loader import llm
from multilevel_rag_fxn import search_semantic
from sentence_transformers import SentenceTransformer

semantic_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Utility: build dynamic prompts
def build_prompt(context_logs, query, fcaps=None, risk=None, tool_type="semantic"):
    base_intro = f"""You are a highly specialized semantic log analyst with FCAPS expertise.
Logs:
{context_logs}

User Query:
{query}

"""
    if tool_type == "semantic":
        task = "Analyze and explain the relationship or diagnosis. If sufficient evidence is missing, make reasoned inferences."
    elif tool_type == "time":
        task = "Identify patterns over time or durations and infer if any critical time-based issues exist."
    elif tool_type == "cluster":
        task = "Detect repeating patterns, grouped behaviors or recurring anomalies."
    elif tool_type == "fcaps":
        task = f"Categorize into FCAPS domain ({fcaps}) and assess risk level ({risk}). Suggest mitigation if risk is high."
    else:
        task = "Analyze and respond clearly."

    closing = "\nAnswer (detailed, actionable, structured if needed):"
    return base_intro + task + closing

# Generalized tool runner
def tool_runner(query, tool_type="semantic", fcaps=None, risk=None):
    rag_results = search_semantic(query=query, model=semantic_model, top_k=5)
    logs = [r[-1] for r in rag_results]
    context_logs = "\n".join([f"- {r[-1]}" for r in rag_results])

    prompt = build_prompt(context_logs, query, fcaps=fcaps, risk=risk, tool_type=tool_type)
    print(f"[Tool:{tool_type}] Logs sent to LLM:\n", context_logs)
    return llm.invoke(prompt)

#  modular tools
semantic_tool = Tool.from_function(
    name="SemanticAnalyzerTool",
    description="Use for causality, correlation, and general analysis.",
    func=lambda q: tool_runner(q, tool_type="semantic")
)

time_tool = Tool.from_function(
    name="TimeRangeLogAnalyzerTool",
    description="Use for time-based analysis of events, durations, anomalies.",
    func=lambda q: tool_runner(q, tool_type="time")
)

fcaps_tool = Tool.from_function(
    name="FCAPSCategorizerTool",
    description="Use for diagnosing faults, configurations, accounting, performance, security issues.",
    func=lambda q: tool_runner(q, tool_type="fcaps")
)

cluster_tool = Tool.from_function(
    name="ClusterPatternTool",
    description="Use for discovering repeated patterns or grouped errors.",
    func=lambda q: tool_runner(q, tool_type="cluster")
)

tools_by_intent = {
    "semantic": semantic_tool,
    "time": time_tool,
    "fcaps": fcaps_tool,
    "cluster": cluster_tool
}




#=================================================================================

# # /RAG/agent_tools.py
# from langchain.tools import Tool
# from mistral_llm_loader import llm
# from multilevel_rag_fxn import search_semantic
# from sentence_transformers import SentenceTransformer

# semantic_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# def semantic_analyzer_fn(query):
#     # Step 1: Retrieve top-k logs using your RAG engine
#     rag_results = search_semantic(query=query, model=semantic_model, top_k=5)
#     context_logs = "\n".join([f"- {log}" for _, _, _, log in rag_results])

#     # Step 2: Inject logs into prompt
#     prompt = f"""You are a semantic log analyst. Explain the relationship or diagnosis for the query below using context or general reasoning.

# Logs:
# {context_logs}

# User Query:
# {query}

# Answer (clear and actionable):"""
    
#     print("Logs sent to LLM:\n", context_logs)
#     return llm.invoke(prompt)

# def time_range_fn(query):
#     rag_results = search_semantic(query=query, model=semantic_model, top_k=5)
#     context_logs = "\n".join([f"- {log}" for _, _, _, log in rag_results])
    
#     prompt = f"""You are a log analyst. Find and summarize patterns based on time or durations in the following query.

# Logs:
# {context_logs}

# User Query:
# {query}

# Answer (clear and actionable):"""
    
#     print("Logs sent to LLM:\n", context_logs)
#     return llm.invoke(prompt)

# def fcaps_tool_fn(query):
#     # Step 1: Retrieve top-k logs using your RAG engine
#     rag_results = search_semantic(query=query, model=semantic_model, top_k=5)
#     context_logs = "\n".join([f"- {log}" for _, _, _, log in rag_results])
    
#     prompt = f"""You are an expert in FCAPS (Fault, Configuration, Accounting, Performance, Security). Analyze this query and determine its nature.

# Logs:
# {context_logs}

# User Query:
# {query}

# Answer(clear and actionable):"""

#     print("Logs sent to LLM:\n", context_logs)
#     return llm.invoke(prompt)

# def cluster_tool_fn(query):
#     rag_results = search_semantic(query=query, model=semantic_model, top_k=5)
#     context_logs = "\n".join([f"- {log}" for _, _, _, log in rag_results])
    
#     prompt = f"""You are a pattern detection engine. Identify repeating log patterns or cluster behavior for the query.

# Logs:
# {context_logs}

# User Query:
# {query}

# Answer(clear and actionable):"""
    
#     print("Logs sent to LLM:\n", context_logs)
#     return llm.invoke(prompt)


# semantic_tool = Tool.from_function(
#     name="SemanticAnalyzerTool",
#     description="Use for questions asking about reasons, cause, correlation",
#     func=semantic_analyzer_fn
# )

# time_tool = Tool.from_function(
#     name="TimeRangeLogAnalyzerTool",
#     description="Use for questions about time ranges, duration, time-based filters",
#     func=time_range_fn
# )

# fcaps_tool = Tool.from_function(
#     name="FCAPSCategorizerTool",
#     description="Use for categorizing logs in FCAPS domain (Fault, Config, etc.)",
#     func=fcaps_tool_fn
# )

# cluster_tool = Tool.from_function(
#     name="ClusterPatternTool",
#     description="Use for detecting repeated patterns or grouped failures",
#     func=cluster_tool_fn
# )

# tools_by_intent = {
#     "semantic": semantic_tool,
#     "time": time_tool,
#     "fcaps": fcaps_tool,
#     "cluster": cluster_tool
# }

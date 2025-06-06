# /RAG/llm_agent.py

from langchain.agents import initialize_agent
from mistral_llm_loader import llm
from agent_tools import tools_by_intent, tool_runner
from query_analyzer import analyze_query
from multilevel_rag_fxn import search_semantic


def fetch_logs(query: str, intent: str, top_k=5):
    if intent in {"semantic", "time", "fcaps", "cluster"}:
        results = search_semantic(query, top_k=top_k)
        return [r[-1] for r in results]  # actual log strings
    return []


def run_llm_agent(query: str):
    subqueries = analyze_query(query)
    all_responses = []
    print(" Subqueries from analyzer:")

    for sq in subqueries:
        print(f" - {sq['query']} (Intent: {sq['intent']} | FCAPS: {sq['fcaps']} | Risk: {sq['risk']})")
        tool = tools_by_intent.get(sq["intent"], tools_by_intent["semantic"])

        matched_logs = fetch_logs(sq["query"], sq["intent"])
        
        # Dynamically inject FCAPS/risk if needed (especially for fcaps tool)
        if sq["intent"] == "fcaps":
            response = tool_runner(sq["query"], tool_type="fcaps",
                                   fcaps=sq["fcaps"], risk=sq["risk"])
        else:
            response = tool.run(sq["query"], logs=matched_logs)

        formatted = f"Intent → {sq['intent'].capitalize()} | FCAPS → {sq['fcaps']} | Risk → {sq['risk']}\n{response}\n"
        all_responses.append(formatted)

    final_response = "\n\n".join(all_responses)
    return final_response




#==============================================================================






# # /RAG/llm_agent.py

# from langchain.agents import initialize_agent
# from mistral_llm_loader import llm
# from agent_tools import tools_by_intent
# from query_analyzer import analyze_query

# def run_llm_agent(query: str):
#     subqueries = analyze_query(query)
#     all_responses = []
#     print(" Subqueries from analyzer:")

#     for sq in subqueries:
#         print(" -", sq["query"])
#         tool = tools_by_intent.get(sq["intent"], tools_by_intent["semantic"])
#         response = tool.run(sq["query"])
#         all_responses.append(f"{sq['intent'].capitalize()} → {response}")

#     return "\n".join(all_responses)

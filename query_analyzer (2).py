# /RAG/query_analyzer.py


import re
import spacy
from transformers import pipeline
from sentence_transformers import SentenceTransformer

nlp = spacy.load("en_core_web_sm")
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
semantic_splitter = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Labels for FCAPS
FCAPS_LABELS = ["Fault", "Configuration", "Accounting", "Performance", "Security"]

# split long queries
def semantic_decompose(query, threshold=0.7):
    sentences = [sent.text.strip() for sent in nlp(query).sents if sent.text.strip()]
    embeddings = semantic_splitter.encode(sentences)
    if len(embeddings) < 2:
        return sentences

    groups = []
    current_group = [sentences[0]]
    for i in range(1, len(sentences)):
        sim = (embeddings[i] @ embeddings[i-1]) / (np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[i-1]))
        if sim < threshold:
            groups.append(" ".join(current_group))
            current_group = []
        current_group.append(sentences[i])
    if current_group:
        groups.append(" ".join(current_group))
    return groups


# FCAPS + Risk detection
def infer_fcaps_and_risk(query: str) -> tuple:
    if not query.strip():
        return ("Unclassified", "Unknown")
    try:
        result = classifier(query, FCAPS_LABELS)
        top_label = result["labels"][0]
        top_score = result["scores"][0]
        risk_level = "High" if top_label == "Fault" and top_score > 0.7 else "Medium" if top_score > 0.5 else "Low"
        return (top_label, risk_level)
    except Exception as e:
        print(" FCAPS risk classification failed:", e)
        return ("Unclassified", "Unknown")

# Intent inference (expanded)
def infer_intent(query: str) -> str:
    keywords = {
        "time": ["between", "after", "before", "since", "duration", "time range"],
        "semantic": ["why", "correlate", "cause", "reason", "analyze"],
        "cluster": ["pattern", "grouped", "recurring", "repeat", "frequent"],
        "fcaps": ["fault", "failure", "configuration", "accounting", "performance", "security"]
    }
    for intent, keys in keywords.items():
        if any(k in query.lower() for k in keys):
            return intent
    return "semantic"

# Main Query Analyzer
def analyze_query(query: str, use_fcaps=True) -> list:
    subqueries = semantic_decompose(query)
    results = []
    for sq in subqueries:
        intent = infer_intent(sq)
        if use_fcaps:
            fcaps, risk = infer_fcaps_and_risk(sq)
        else:
            fcaps, risk = ("Unclassified", "Unknown")
        results.append({
            "query": sq,
            "intent": intent,
            "fcaps": fcaps,
            "risk": risk
        })
    return results


#=============================================================================


# import re
# import spacy
# from transformers import pipeline

# nlp = spacy.load("en_core_web_sm")
# classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# def decompose_query(query: str) -> list:
#     """
#     Splits complex queries using logical conjunctions only when they are separate words.
#     Falls back to spaCy sentence segmentation if needed.
#     """
#     # Use \b to match word boundaries around 'and' and 'then'
#     parts = re.split(r'\b(?:and|then)\b|,', query, flags=re.IGNORECASE)
#     clean_parts = [p.strip() for p in parts if len(p.strip().split()) > 3]

#     if clean_parts:
#         return clean_parts

#     # fallback: spaCy
#     doc = nlp(query)
#     return [sent.text.strip() for sent in doc.sents if len(sent.text.strip().split()) > 3]


# def infer_fcaps(query: str, use_fcaps: bool = True) -> str:
#     """
#     Classifies query into FCAPS domain using zero-shot classification.
#     """
#     if not use_fcaps:
#         return "Unclassified"

#     labels = ["Fault", "Configuration", "Accounting", "Performance", "Security"]
#     try:
#         result = classifier(query, labels)
#         return result["labels"][0]
#     except Exception as e:
#         print(" FCAPS classification failed:", e)
#         return "Unclassified"


# def infer_intent(query: str) -> str:
#     """
#     Simple rule-based intent inference (optional).
#     """
#     keywords = {
#         "time": ["between", "after", "before", "since"],
#         "semantic": ["why", "correlate", "cause", "reason"],
#         "cluster": ["pattern", "grouped", "recurring", "repeat"],
#     }
#     for intent, keys in keywords.items():
#         if any(k in query.lower() for k in keys):
#             return intent
#     return "semantic"


# def analyze_query(query: str, use_fcaps=True) -> list:
#     """
#     Returns a list of subquery dictionaries with optional FCAPS and intent tagging.
#     """
#     subqueries = decompose_query(query)
#     results = []

#     for sq in subqueries:
#         results.append({
#             "query": sq,
#             "intent": infer_intent(sq),
#             "fcaps": infer_fcaps(sq, use_fcaps=use_fcaps)
#         })

#     return results

#============================================================================
import faiss
import numpy as np
import pandas as pd
import os
import pickle
import re
from sentence_transformers import CrossEncoder
from datetime import datetime
import dateutil.parser
from sentence_transformers import SentenceTransformer

RAG_FOLDER = "/workspace/FCAPS/RAG"
with open(os.path.join(RAG_FOLDER, "log_mapping.pkl"), "rb") as f:
    log_mapping = pickle.load(f)

metadata_df = pd.read_csv(os.path.join(RAG_FOLDER, "metadata_with_clusters.csv"))
centroids = np.load(os.path.join(RAG_FOLDER, "cluster_centroids.npy"))

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
semantic_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Dataset keyword mapping for fallback
DATASET_KEYWORDS = {
    "android": [
            "android", "intent", "receiver", "broadcast", "systemui", "bootcompleted", "fullscreenstackvis", "dockedstackvis", "setsystemuivisibility", "powermanagerservice", "wksummary", "wakefulness", "recentsactivity", "manual", "auto"
        ],
        
        
        "linux": [
            "linux", "ssh", "sshd", "pam_unix", "authentication", "failure", "nodevssh", "connection",  "session", "unknown", "rhost", "ruser", "logname", "sudo", "login", "access denied"
        ],
        
        
        "openstack": [
            "openstack","osapi_compute", "instance", "server", "nova", "http", "osapi_compute", "virt", "libvirt", "lifecycle", "imagecache", "instance", "server", "rmcommunicator", "lease", "renewer", "client", "taskattempt","compute", "status", "api", "vm", "flavor"
        ],
        
        
        "hadoop": [
            "hadoop", "yarn",    "rmcontainerallocator", "mapreduce", "resourcemanager", "nodemanager",
            "asyncdispatcher", "event", "handler", "transitioned"
        ],
        
        
        "hdfs": [
            "hdfs","fsdataset", "datanode", "dataxceiver", "packetresponder", "blk_", "addstoredblock", "blockmap", "received", "deleted", "verifying", "termination"
        ]
}


def normalize(vec):
    return vec / np.linalg.norm(vec)

# #  Remove near-duplicate logs
# def deduplicate_logs(logs, threshold=0.9):
#     texts = [log[-1] for log in logs] 
#     embeddings = semantic_model.encode(texts, convert_to_numpy=True)
#     norms = np.linalg.norm(embeddings, axis=1)
#     filtered_logs = []
#     seen = []

#     for idx, emb in enumerate(embeddings):
#         if all(np.dot(emb, seen_emb) / (np.linalg.norm(emb) * np.linalg.norm(seen_emb)) < threshold for seen_emb in seen):
#             filtered_logs.append(logs[idx])
#             seen.append(emb)

#     return filtered_logs

# Infer datasets from query keywords
def infer_dataset(query):
    query = query.lower()
    scores = {ds: 0 for ds in DATASET_KEYWORDS}
    for ds, keywords in DATASET_KEYWORDS.items():
        scores[ds] = sum(1 for k in keywords if k in query)

    filtered = [k for k, v in scores.items() if v > 0]
    if filtered:
        return [f"{ds.capitalize()}_2k_cleaned.csv" for ds in filtered]
    return metadata_df["dataset"].unique().tolist()


def extract_date_range(query):
    date_matches = re.findall(r"(\\d{1,2}\\s*\\w+\\s*\\d{4}|\\d{4}-\\d{2}-\\d{2})", query)
    if len(date_matches) == 1:
        try:
            dt = dateutil.parser.parse(date_matches[0], fuzzy=True)
            return dt.date(), dt.date()
        except: pass
    elif len(date_matches) >= 2:
        try:
            d1 = dateutil.parser.parse(date_matches[0], fuzzy=True).date()
            d2 = dateutil.parser.parse(date_matches[1], fuzzy=True).date()
            return min(d1, d2), max(d1, d2)
        except: pass
    return None, None


# Main semantic retrieval
def search_semantic(query, model=None, top_k=5, rerank=True, multi_hop=True):
    if model is None:
        model = semantic_model

    query_embedding = normalize(model.encode([query], convert_to_numpy=True))

    # 1. Infer datasets
    inferred_datasets = infer_dataset(query)
    subset_df = metadata_df[metadata_df["dataset"].isin(inferred_datasets)]

    start_date, end_date = extract_date_range(query)
    if start_date:
        subset_df = subset_df.copy()
        subset_df["timestamp"] = pd.to_datetime(subset_df["timestamp"]).dt.date
        subset_df = subset_df[(subset_df["timestamp"] >= start_date) & (subset_df["timestamp"] <= end_date)]

    if subset_df.empty:
        print(" No matching dataset found, falling back to all logs")
        subset_df = metadata_df.copy()

    # Literal match fallback (match against logs from faiss_index)
    query_lower = query.lower()
    matched = []
    for row in subset_df.itertuples():
        log_index = row.faiss_index
        dataset, row_idx, log_content = log_mapping[log_index]
        if query_lower in log_content.lower():
            matched.append((dataset, row_idx, 1.0, log_content))

    if matched:
        print(f" Found {len(matched)} exact literal matches — skipping vector search")
        return matched[:top_k]

    # 2. Cluster selection
    similarities = np.dot(centroids, query_embedding.T)
    cluster_id = int(np.argmax(similarities))

    cluster_subset = subset_df[subset_df["cluster"] == cluster_id]
    if cluster_subset.empty:
        cluster_subset = subset_df

    cluster_indices = cluster_subset["faiss_index"].astype(np.int64).values

    index_flat = faiss.read_index(os.path.join(RAG_FOLDER, "logs.index"))
    cluster_embeddings = np.array([index_flat.reconstruct(int(i)) for i in cluster_indices])
    faiss.normalize_L2(cluster_embeddings)

    # 3. Search
    hnsw = faiss.IndexHNSWFlat(cluster_embeddings.shape[1], 32)
    hnsw.hnsw.efSearch = 128
    hnsw.add(cluster_embeddings.astype('float32'))

    faiss.normalize_L2(query_embedding)
    _, indices = hnsw.search(query_embedding.astype('float32'), top_k * 5)

    results = []
    for i in indices[0]:
        if i < len(cluster_indices):
            log_index = cluster_indices[i]
            dataset, row_idx, log_content = log_mapping[log_index]
            results.append((dataset, row_idx, 0.0, log_content))

    if not results:
        fallback = cluster_subset.sample(n=min(top_k, len(cluster_subset)))
        return [(row["dataset"], row["row_index"], 0.0, row["log"]) for _, row in fallback.iterrows()]

    # 4. Rerank
    if rerank and reranker:
        texts = [(query, r[3]) for r in results]
        scores = reranker.predict(texts)
        results = sorted(zip(results, scores), key=lambda x: x[1], reverse=True)
        results = [(d, r, conf, round(s, 4), log) for ((d, r, conf, log), s) in
                   results]


    # # 5. Deduplicate
    # results = deduplicate_logs(results, threshold=0.95)

    # 6. Multi-hop retrieval (optional)
    if multi_hop:
        hints = " ".join([str(r[3]) for r in results[:3]])
        hint_query = query + " " + hints[:512]
        secondary_results = search_semantic(hint_query, model, top_k=top_k, rerank=rerank, multi_hop=False)
        results += secondary_results

    return results[:top_k]








# ===================================================================

# # rag_fxn.py with Multi-Level RAG logic + Dataset Filter + Keyword Filter

# import faiss
# import numpy as np
# import pandas as pd
# import os
# import pickle
# import re
# from sentence_transformers import CrossEncoder
# from datetime import datetime

# RAG_FOLDER = "/workspace/FCAPS/RAG"

# # Load metadata and mapping
# with open(os.path.join(RAG_FOLDER, "log_mapping.pkl"), "rb") as f:
#     log_mapping = pickle.load(f)

# metadata_df = pd.read_csv(os.path.join(RAG_FOLDER, "metadata_with_clusters.csv"))
# centroids = np.load(os.path.join(RAG_FOLDER, "cluster_centroids.npy"))

# # reranker
# reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# def infer_dataset(query: str) -> list:
#     query = query.lower()
#     dataset_keywords = {
#         "android": [
#             "android", "intent", "receiver", "broadcast", "systemui", "bootcompleted", "fullscreenstackvis", "dockedstackvis", "setsystemuivisibility", "powermanagerservice", "wksummary", "wakefulness", "recentsactivity", "manual", "auto"
#         ],
        
        
#         "linux": [
#             "linux", "ssh", "sshd", "pam_unix", "authentication", "failure", "nodevssh", "connection",  "session", "unknown", "rhost", "ruser", "logname", "sudo", "login", "access denied"
#         ],
        
        
#         "openstack": [
#             "openstack","osapi_compute", "instance", "server", "nova", "http", "osapi_compute", "virt", "libvirt", "lifecycle", "imagecache", "instance", "server", "rmcommunicator", "lease", "renewer", "client", "taskattempt","compute", "status", "api", "vm", "flavor"
#         ],
        
        
#         "hadoop": [
#             "hadoop", "yarn",    "rmcontainerallocator", "mapreduce", "resourcemanager", "nodemanager",
#             "asyncdispatcher", "event", "handler", "transitioned"
#         ],
        
        
#         "hdfs": [
#             "hdfs","fsdataset", "datanode", "dataxceiver", "packetresponder", "blk_", "addstoredblock", "blockmap", "received", "deleted", "verifying", "termination"
#         ]
#     }

#     scores = {ds: 0 for ds in dataset_keywords}
#     for ds, keywords in dataset_keywords.items():
#         scores[ds] = sum(1 for k in keywords if k in query)
    
#     filtered = [k for k, v in scores.items() if v > 0]
#     if filtered:
#         return [f"{ds.capitalize()}_2k_cleaned.csv" for ds in filtered]
#     return metadata_df["dataset"].unique().tolist() 



# def search_semantic(query, model, top_k=5, metric='cosine', rerank=True):
#     if model is None:
#         raise ValueError("A sentence transformer model must be provided")

#     query_embedding = model.encode([query], convert_to_numpy=True)
#     query_embedding = query_embedding / np.linalg.norm(query_embedding)

#     # Step 1: Infer datasets
#     inferred_datasets = infer_dataset(query)
#     print(f"Inferred datasets: {inferred_datasets}")

#     # Step 2: Cluster selection
#     if metric == 'cosine':
#         similarities = np.dot(centroids, query_embedding.T)
#     elif metric == 'euclidean':
#         similarities = -np.linalg.norm(centroids - query_embedding, axis=1)
#     elif metric == 'manhattan':
#         similarities = -np.sum(np.abs(centroids - query_embedding), axis=1)
#     else:
#         raise ValueError("Unsupported metric")

#     cluster_id = int(np.argmax(similarities))
#     print(f" Selected cluster ID: {cluster_id}")

#     subset_df = metadata_df[(metadata_df["dataset"].isin(inferred_datasets))]

#     # Step 3: Try restricting to cluster
#     cluster_subset = subset_df[subset_df["cluster"] == cluster_id]

#     if cluster_subset.empty:
#         print(" No logs in matching cluster, falling back to dataset-wide search")
#         cluster_subset = subset_df

#     if cluster_subset.empty:
#         print(" No matching logs in any dataset — fallback failed")
#         return []

#     cluster_indices = cluster_subset["faiss_index"].astype(np.int64).values

#     index_flat = faiss.read_index(os.path.join(RAG_FOLDER, "logs.index"))
#     cluster_embeddings = np.array([index_flat.reconstruct(int(i)) for i in cluster_indices])
#     faiss.normalize_L2(cluster_embeddings)
#     faiss.normalize_L2(query_embedding)

#     hnsw = faiss.IndexHNSWFlat(cluster_embeddings.shape[1], 32)
#     hnsw.hnsw.efSearch = 128
#     hnsw.add(cluster_embeddings.astype('float32'))

#     _, indices = hnsw.search(query_embedding.astype('float32'), top_k * 3)

#     results = []
#     for i in indices[0]:
#         if i < len(cluster_indices):
#             log_index = cluster_indices[i]
#             dataset, row_idx, log_content = log_mapping[log_index]
#             results.append((dataset, row_idx, log_content))

#     if not results:
#         print(" No results from FAISS — fallback to random logs from dataset")
#         fallback = cluster_subset.sample(n=min(top_k, len(cluster_subset)))
#         return [(row["dataset"], row["row_index"], 0.0, row["log"]) for _, row in fallback.iterrows()]

#     # Step 4: Rerank using cross-encoder
#     if rerank and reranker:
#         texts = [(query, r[2]) for r in results]
#         scores = reranker.predict(texts)
#         reranked = sorted(zip(results, scores), key=lambda x: x[1], reverse=True)
#         top_reranked = reranked[:top_k]
#         return [(d, r, round(s, 4), log) for ((d, r, log), s) in top_reranked]

#     return [(d, r, 0.0, log) for (d, r, log) in results[:top_k]]


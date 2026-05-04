# Week 09 — Retrieval-Augmented Generation (RAG) and Large Language Models

## Overview

This lab explores how **Retrieval-Augmented Generation (RAG)** extends Large Language Models (LLMs) by supplying external knowledge at inference time. You will build every component of a RAG pipeline from scratch: chunking documents, embedding and indexing them in a vector store, retrieving relevant passages, and augmenting a prompt with that context before generation. You will also evaluate retrieval quality rigorously and study how poor retrieval leads to hallucination.

## Learning Goals

- Understand the **knowledge limitations** of LLMs (cutoff, hallucination) and why RAG addresses them.
- Implement **document chunking** strategies: fixed-size with overlap and sentence-boundary.
- Build a **TF-IDF cosine similarity vector store** for nearest-neighbour retrieval.
- Evaluate retrieval using **Precision@k**, **Recall@k**, and **Mean Reciprocal Rank (MRR)**.
- Construct a complete **RAG pipeline**: retrieve → augment prompt → generate.
- Analyse **hallucination** and implement a retrieval-score threshold for abstention.
- Compare **RAG vs fine-tuning** and understand advanced indexing strategies.

## Setup

```bash
pip install sentence-transformers faiss-cpu transformers torch numpy pandas matplotlib scikit-learn
```

---

## Part 1 — LLMs and the Knowledge Problem

LLMs encode world knowledge in their parameters during pre-training but face two fundamental limitations: a **knowledge cutoff** (no information about events after training) and **hallucination** (confident generation of incorrect facts). RAG mitigates both by retrieving relevant external documents at query time and injecting them into the prompt.

### Task 1A — Identifying Hallucination-Prone Queries

Classify eight queries as Low Risk or High Risk for hallucination. High-risk queries require timely, private, or highly specific factual knowledge the model is unlikely to have seen.

### Discussion Questions

1. What is the difference between **parametric knowledge** (stored in weights) and **non-parametric knowledge** (retrieved at inference time)?
2. Name two scenarios where RAG is preferred over fine-tuning, and two where fine-tuning is preferred.
3. What is **context stuffing** and why is it not a scalable solution?

---

## Part 2 — Document Chunking

Before embedding, documents must be split into chunks small enough to be meaningfully retrieved. Chunk size and overlap control the precision-recall tradeoff in retrieval.

### Coding Task A — Fixed-size Chunking with Overlap

Implement a sliding-window word-level chunker. Given `chunk_size` and `overlap`, advance by `chunk_size - overlap` words per step.

### Coding Task B — Sentence-boundary Chunking

Split on sentence-ending punctuation, then apply a sliding window over sentences with configurable overlap.

### Coding Task C — Build a Chunked Corpus

Apply fixed-size chunking to all five documents in the knowledge base and produce a flat list of chunk dicts with `chunk_id`, `doc_id`, `title`, and `text`.

### Discussion Questions

1. What effect does increasing overlap have on retrieval quality and storage cost?
2. Why might sentence-boundary chunking outperform fixed-size chunking for certain query types?
3. Describe a case where very small chunks hurt retrieval and a case where very large chunks hurt retrieval.

---

## Part 3 — Embeddings and Vector Store

Each chunk is mapped to a dense (or sparse) vector and stored. Retrieval finds the nearest vectors to a query vector.

### Coding Task A — TF-IDF Vector Store

Implement a `TFIDFVectorStore` class that:
- Fits a `TfidfVectorizer` on all chunk texts.
- Stores the resulting sparse matrix.
- Implements a `query(text, k)` method that transforms the query and returns the top-k chunks by cosine similarity.

### Discussion Questions

1. Why is cosine similarity preferred over Euclidean distance for TF-IDF vectors?
2. What is the main limitation of TF-IDF compared to dense sentence embeddings? Give a concrete failure example.
3. What is an **inverted index** and how does it speed up sparse retrieval?

---

## Part 4 — Retrieval Evaluation

### Coding Task A — Precision@k and Recall@k

Implement `precision_at_k` (fraction of top-k results that are relevant) and `recall_at_k` (fraction of all relevant docs found in top-k).

### Coding Task B — Mean Reciprocal Rank

Implement `reciprocal_rank` (returns 1/rank of the first relevant result, 0 if none) and `mean_reciprocal_rank` averaged across all queries.

### Coding Task C — P@k and R@k Curves

Plot mean Precision@k and Recall@k for k = 1 … 5.

### Discussion Questions

1. Explain the precision-recall tradeoff as k increases.
2. Why is MRR useful when only the **first** relevant result matters?
3. How would you handle multi-document relevance in evaluation?

---

## Part 5 — The RAG Pipeline

### Coding Task A — Prompt Template Construction

Implement `build_rag_prompt(query, retrieved_chunks, max_context_words)`. Number each chunk in the context block, truncate at `max_context_words`, and fill the template:

```
You are a helpful NLP teaching assistant. Use only the provided context
to answer the question. If the answer is not in the context, say "I don't know".

Context:
{context}

Question: {question}

Answer:
```

### Coding Task B — Simulated Extractive Generator

Implement `rag_answer(query, vector_store, k)` that calls retrieve → augment → generate. The generator is a TF-IDF-scored extractive function (replace with a real LLM API when available).

### Optional Extension

Swap in a real LLM using the Anthropic SDK:

```python
import anthropic
client = anthropic.Anthropic(api_key="YOUR_KEY")
response = client.messages.create(
    model="claude-haiku-4-5-20251001",
    max_tokens=256,
    messages=[{"role": "user", "content": prompt}]
)
```

### Discussion Questions

1. What does the prompt template communicate to the model? Why is "use only the context" important?
2. How would you modify the pipeline for queries requiring information from multiple documents?
3. What is **re-ranking** and why is it useful as a second retrieval stage?

---

## Part 6 — Hallucination Analysis and Context Quality

### Coding Task A — Out-of-scope Queries

Run the RAG pipeline on queries with no relevant document in the knowledge base. Observe that the retrieval score is low and the generated answer is unreliable.

### Coding Task B — Retrieval Score Threshold

Implement `rag_with_threshold(query, vector_store, k, threshold)`. If the top cosine similarity score is below `threshold`, return a canned "I don't know" response instead of generating.

### Coding Task C — Effect of k on Answer Quality

Measure word-overlap F1 between generated answers and reference answers for k = 1 … 5. Analyse whether more retrieved context always helps.

### Discussion Questions

1. Distinguish between **intrinsic hallucination** (contradicts context) and **extrinsic hallucination** (unverifiable from context).
2. What are the risks of setting the abstention threshold too high vs too low?
3. How does k affect hallucination risk?
4. Describe two approaches other than score thresholding that reduce hallucination.

---

## Part 7 — Advanced RAG Concepts (Discussion)

### Hybrid Retrieval and Reciprocal Rank Fusion

Compute RRF scores by hand for a sparse-ranker and a dense-ranker result list (k=60). Produce the fused ranking.

### Indexing Strategies

Complete a comparison table for: Flat exact search, IVF, HNSW, and Product Quantisation — one advantage and one disadvantage each.

### RAG vs Fine-Tuning Trade-offs

Fill in a comparison table across: up-to-date knowledge, transparency, inference cost, domain style, hallucination risk, and latency.

---

## Submission

- Complete all `TODO` sections in the notebook.
- Answer all written questions in the markdown cells.
- Include in your submission:
  - The **P@k / R@k curve plot** from Part 4 (`retrieval_pk_rk.png`)
  - The **retrieval evaluation table** from Part 4 with mean P@3, R@3, and MRR
  - The **threshold abstention table** from Part 6B
  - The **k vs F1 table** from Part 6C
- Submit your completed `.ipynb` file as instructed.

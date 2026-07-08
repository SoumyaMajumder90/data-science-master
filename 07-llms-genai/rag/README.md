# RAG (Retrieval-Augmented Generation)

> A pattern that grounds an LLM's answer in **retrieved external documents** instead of relying only on parametric memory — reducing hallucination and enabling fresh, private, or domain-specific knowledge.

| | |
|---|---|
| **Category** | LLMs & Generative AI |
| **Difficulty** | 🟨 Intermediate |
| **Prerequisites** | [Embeddings & Similarity Search](../embeddings-and-similarity-search/), [Vector Databases](../vector-databases/), [Transformer Architecture](../transformer-architecture/) |
| **Cheatsheet** | [cheatsheet.md](./cheatsheet.md) |

---

## 1. Intuition
An LLM only "knows" what was in its training data, frozen at its cutoff. RAG gives it an open book: at query time you **retrieve** the most relevant passages from your own corpus and paste them into the prompt, so the model answers *from the sources* rather than from memory. Think open-book exam vs closed-book.

## 2. Formal definition / Key concepts
Two stages:
1. **Retrieval** — embed the query, find nearest document chunks in a vector store (semantic search), optionally re-rank.
2. **Generation** — feed retrieved chunks + the query to the LLM as context; it synthesizes a grounded answer, ideally with citations.

Key components: **chunking**, **embedding model**, **vector index**, **retriever**, optional **re-ranker**, and the **generator** (LLM).

## 3. Math
Retrieval scores chunks by similarity to the query embedding, typically cosine:
$$\text{sim}(q, d) = \frac{\mathbf{e}_q \cdot \mathbf{e}_d}{\lVert \mathbf{e}_q \rVert \, \lVert \mathbf{e}_d \rVert}$$

Return the top-$k$ chunks by score; approximate nearest-neighbour indexes (HNSW, IVF) make this fast at scale.

## 4. How it works
1. **Offline (indexing):** split documents into chunks → embed each → store vectors + metadata in a vector DB.
2. **Online (query):** embed the query → retrieve top-$k$ similar chunks → (optionally re-rank) → build a prompt that includes them → LLM generates the answer with citations.

## 5. When to use / When not to
- ✅ Answers must reflect private, changing, or post-cutoff knowledge.
- ✅ You need source attribution / auditability.
- ✅ Cheaper and faster to update than fine-tuning.
- ❌ The knowledge is small and static → just put it in the prompt.
- ❌ The task needs new *skills/behavior* rather than *facts* → fine-tuning fits better.

## 6. Common pitfalls & gotchas
- **Chunking dominates quality** — too big dilutes relevance, too small loses context. Overlap helps.
- **Retrieval failure ≠ generation failure** — evaluate them separately (retrieval recall vs answer faithfulness).
- **Lost in the middle** — LLMs attend less to context buried in the center of a long prompt.
- Embedding model mismatch (query vs docs) tanks recall; use the same model for both.
- Grounded ≠ correct — the model can still misread retrieved text; require citations and verify.

## 7. Code
```python
# Sketch: embed query, retrieve, then generate
q_vec = embed(query)
chunks = vector_db.search(q_vec, top_k=5)          # semantic retrieval
context = "\n\n".join(c.text for c in chunks)
prompt = f"Answer using ONLY the context. Cite sources.\n\nContext:\n{context}\n\nQ: {query}"
answer = llm.generate(prompt)
```

## 8. Interview / viva questions
- Q: RAG vs fine-tuning — when each?
  - A: RAG for changing/private *facts* and attribution; fine-tuning for new *behavior/style/format*. They're complementary.
- Q: Your RAG gives wrong answers. How do you debug?
  - A: Separate the stages — check whether retrieval returned the right chunks (recall) before blaming generation (faithfulness).
- Q: Why chunk overlap?
  - A: To avoid splitting a relevant fact across a boundary and losing it at retrieval time.

## 9. References
- Lewis et al. (2020) — "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks."
- Liu et al. (2023) — "Lost in the Middle: How Language Models Use Long Contexts."

---
> _Status: 🟢 done (example note)._

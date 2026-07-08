# RAG — Cheatsheet

> Quick recall. For depth see [README.md](./README.md).

**TL;DR:** Retrieve relevant docs → stuff into the prompt → LLM answers grounded in them. Open-book LLM.

## Key formulas
- Cosine similarity: $\text{sim}(q,d) = \frac{\mathbf{e}_q\cdot\mathbf{e}_d}{\lVert e_q\rVert\lVert e_d\rVert}$
- Retrieve top-$k$ by similarity (HNSW / IVF for speed).

## Must-know facts
- Two stages: **retrieval** + **generation** — evaluate separately.
- Fixes hallucination on *facts*; not a substitute for *skills* → that's fine-tuning.
- Same embedding model for query and docs.
- "Lost in the middle": models underuse mid-prompt context.

## Quick decisions
| Situation | Do this |
|---|---|
| Facts change often / are private | RAG |
| Need new style or behavior | Fine-tune |
| Poor answers | Check retrieval recall first |
| Facts split across chunks | Add chunk overlap |

## Common mistakes
- Blaming the LLM when retrieval missed the chunk.
- Bad chunk size (too big dilutes, too small loses context).
- Different embedding models for query vs docs.

## One-liner code
```python
chunks = db.search(embed(query), top_k=5); llm.generate(prompt_with(chunks))
```

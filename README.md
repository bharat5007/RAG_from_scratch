# RAG from Scratch

A hands-on implementation of Retrieval-Augmented Generation (RAG) techniques, built notebook-by-notebook from first principles. Covers the full pipeline from document ingestion to final answer generation, with deep dives into each component.

## Repository Structure

| Notebook | What it covers |
|---|---|
| [rag.ipynb](rag.ipynb) | Baseline RAG — indexing, retrieval, generation |
| [chunking.ipynb](chunking.ipynb) | Chunking strategies: fixed-size, sentence/paragraph, recursive, semantic, parent-child |
| [indexing.ipynb](indexing.ipynb) | Advanced indexing: multi-representation (summaries), RAPTOR, ColBERT |
| [searching.ipynb](searching.ipynb) | Retrieval methods: BM25, dense retrieval, hybrid search + RRF, HyDE, cross-encoder re-ranking |
| [query_transformation.ipynb](query_transformation.ipynb) | Query rewriting: decomposition, step-back prompting, HyDE |
| [routing.ipynb](routing.ipynb) | Logical & semantic routing, query structuring for metadata filters |
| [full_pipeline.ipynb](full_pipeline.ipynb) | End-to-end pipeline: ingestion → cleaning → chunking → indexing → transformation → retrieval |

## Techniques Covered

**Chunking**
- Fixed-size with overlap
- Sentence / paragraph splitting
- Recursive character splitting (LangChain)
- Semantic chunking
- Parent-child chunking

**Indexing**
- Multi-representation indexing (index summaries, retrieve full docs)
- RAPTOR — hierarchical summarization tree with UMAP + Gaussian Mixture clustering
- ColBERT via RAGatouille (late interaction retrieval)

**Searching / Retrieval**
- BM25 (custom implementation + `rank_bm25`)
- Dense retrieval with bi-encoders (`sentence-transformers`)
- Hybrid search with Reciprocal Rank Fusion (RRF)
- HyDE — Hypothetical Document Embeddings
- Cross-encoder re-ranking

**Query Transformation**
- Multi-query decomposition
- Step-back prompting
- HyDE query expansion

**Routing**
- Logical routing (rule-based)
- Semantic routing (embedding similarity)
- Query structuring for metadata filters

## Stack

- **LLM**: Groq (`llama-3.3-70b-versatile`)
- **Embeddings**: HuggingFace (`all-MiniLM-L6-v2`)
- **Vector stores**: ChromaDB, Weaviate
- **Frameworks**: LangChain, RAGatouille (ColBERT)
- **Other**: UMAP, scikit-learn, sentence-transformers

## Setup

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install langchain langchain-community langchain-groq langchain-huggingface
pip install chromadb weaviate-client sentence-transformers rank-bm25
pip install ragatouille umap-learn scikit-learn
pip install python-dotenv bs4
```

Create a `.env` file with your API keys:

```
GROQ_API_KEY=your_groq_api_key
```

Then open any notebook and run cells in order.

## Reference

Notebooks follow the structure from [RAG from scratch](https://github.com/langchain-ai/rag-from-scratch) by LangChain, with custom implementations and extensions.

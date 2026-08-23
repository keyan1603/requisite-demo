"""
05 - RAG: retrievers, hybrid search, reranking, and context compression.

Run with:
    python demos/05_rag.py
"""

from _shared import gemini_api_key, make_agent, make_ai
from requisite.capabilities import default_registry as capabilities
from requisite.rag import BM25Retriever, HybridRetriever, LLMContextCompressor, LLMReranker, Retriever
from requisite.rag.embeddings import GeminiEmbeddingProvider
from requisite.rag.vectorstores import InMemoryVectorStore

DOCS = [
    "Requisite is a provider-agnostic Python framework for building AI applications.",
    "Paris is the capital of France. The Eiffel Tower was completed in 1889.",
    "The Model Context Protocol (MCP) lets AI applications connect to external tools and data sources.",
    "The mitochondria is the powerhouse of the cell.",
    "REQ-4471 tracks the outstanding refund for order 88213.",
    "The mitochondria produces ATP through oxidative phosphorylation, "
    "which powers nearly all cellular activity.",
]


def dense_retrieval_and_capability() -> None:
    print("=== Retriever (dense/embedding-based) ===")
    retriever = Retriever(
        embedding_provider=GeminiEmbeddingProvider(api_key=gemini_api_key()),
        vector_store=InMemoryVectorStore(),
    )
    retriever.add_texts(DOCS)

    results = retriever.retrieve("What is MCP?", top_k=1)
    for scored_chunk in results:
        print(f"  [{scored_chunk.score:.3f}] {scored_chunk.chunk.text}")

    # Exposed as a capability -- an agent can't tell this apart from a
    # native tool or an MCP-backed one.
    capabilities.register("knowledge_base", retriever.as_tool())
    agent = make_agent("Assistant")
    agent.requires("knowledge_base")
    print("\nAgent using the knowledge_base capability:")
    print(agent.run("What is MCP, according to the knowledge base?").content)


def bm25_example() -> None:
    print("\n=== BM25Retriever (keyword-only, zero dependency) ===")
    retriever = BM25Retriever()
    retriever.add_texts(DOCS)
    results = retriever.retrieve("REQ-4471 refund", top_k=1)
    print(f"  [{results[0].score:.3f}] {results[0].chunk.text}")

    # Regression check for a bug fixed in ADR-0031: top_k=0 must return
    # nothing, not silently fall back to a default count.
    assert retriever.retrieve("REQ-4471 refund", top_k=0) == []
    print("  top_k=0 correctly returns [] (ADR-0031 regression check passed)")


def hybrid_and_rerank_example() -> None:
    print("\n=== HybridRetriever (dense + BM25, fused via RRF) ===")
    retriever = HybridRetriever(
        embedding_provider=GeminiEmbeddingProvider(api_key=gemini_api_key()),
        vector_store=InMemoryVectorStore(),
    )
    retriever.add_texts(DOCS)

    keyword_hit = retriever.retrieve("REQ-4471 refund status", top_k=1)[0]
    print(f"  keyword query  -> [{keyword_hit.score:.3f}] {keyword_hit.chunk.text}")

    semantic_hit = retriever.retrieve("Where is the famous Parisian iron tower?", top_k=1)[0]
    print(f"  semantic query -> [{semantic_hit.score:.3f}] {semantic_hit.chunk.text}")

    print("\n=== LLMReranker (listwise, one structured-output call) ===")
    query = "what powers a biological cell?"
    candidates = retriever.retrieve(query, top_k=5)
    reranker = LLMReranker(ai=make_ai(provider="gemini"))
    reranked = reranker.rerank(query, candidates, top_k=2)
    print(f"  [{reranked[0].score:.1f}] {reranked[0].chunk.text}")

    print("\n=== LLMContextCompressor (shrinks each passage to what's relevant) ===")
    compressor = LLMContextCompressor(ai=make_ai(provider="gemini"))
    compressed = compressor.compress(query, reranked)
    for scored_chunk in compressed:
        print(f"  [{scored_chunk.score:.1f}] {scored_chunk.chunk.text}")


def main() -> None:
    dense_retrieval_and_capability()
    bm25_example()
    hybrid_and_rerank_example()


if __name__ == "__main__":
    main()

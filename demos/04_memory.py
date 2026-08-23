"""
04 - Conversation memory: persisting history across separate run() calls,
with three interchangeable backends.

Run with:
    python demos/04_memory.py
"""

import tempfile
from pathlib import Path

from _shared import gemini_api_key, make_agent
from requisite.memory import InProcessMemory, VectorMemory
from requisite.memory.sqlite import SQLiteMemory
from requisite.rag.embeddings.gemini import GeminiEmbeddingProvider
from requisite.rag.vectorstores import InMemoryVectorStore


def in_process_example() -> None:
    print("=== InProcessMemory (lost on restart, zero setup) ===")
    memory = InProcessMemory()
    agent = make_agent("Assistant", memory=memory, session_id="user-42")

    print(agent.run("My favorite language is Python.").content)
    print(agent.run("What's my favorite language?").content)


def sqlite_example() -> None:
    print("\n=== SQLiteMemory (persists across restarts) ===")
    db_path = str(Path(tempfile.mkdtemp()) / "conversations.db")

    first_run = make_agent("Assistant", memory=SQLiteMemory(db_path=db_path), session_id="user-42")
    print(first_run.run("My favorite color is teal.").content)

    # A brand new Agent + SQLiteMemory instance, pointed at the same file --
    # simulates the process restarting. It still recalls the earlier turn.
    second_run = make_agent("Assistant", memory=SQLiteMemory(db_path=db_path), session_id="user-42")
    print(second_run.run("What's my favorite color?").content)


def vector_memory_example() -> None:
    print("\n=== VectorMemory (semantic recall, not just chronological) ===")
    memory = VectorMemory(
        embedding_provider=GeminiEmbeddingProvider(api_key=gemini_api_key()),
        vector_store=InMemoryVectorStore(),
    )
    agent = make_agent("Assistant", memory=memory, session_id="user-42")

    agent.run("My favorite color is teal.")
    agent.run("My favorite food is ramen.")
    agent.run("I'm planning a trip to Japan next spring.")

    print("Chronological history (memory.load):")
    for message in memory.load("user-42"):
        print(f"  [{message.role.value}] {message.content}")

    # load_relevant() is an explicit, opt-in call -- not part of Agent's
    # own contract -- for pulling the most semantically relevant past
    # turns for a query, distinct from the full chronological history.
    print("\nSemantically relevant to 'What should I eat?':")
    for message in memory.load_relevant("user-42", "What should I eat?", top_k=1):
        print(f"  [{message.role.value}] {message.content}")


def main() -> None:
    in_process_example()
    sqlite_example()
    vector_memory_example()


if __name__ == "__main__":
    main()

"""
Phase 3 Multi-User Security and Data Isolation Test Suite.
Verifies:
1. Token verification and auth configurations
2. Strict database separation between multiple distinct users
3. Isolated FAISS vector stores per user
4. Markdown & PDF export engines
"""

import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from database import Database
from auth.firebase_config import get_firebase_web_config
from auth.firebase_admin_init import verify_token
from vector_store import VectorStore
from rag import RAGPipeline
from export import export_to_markdown, export_to_pdf

def test_auth():
    print("\n--- [1] Testing Auth & Token Verification ---")
    config = get_firebase_web_config()
    print(f"[PASS] Loaded Web Config with Project ID: {config['projectId']}")
    assert "apiKey" in config

    # Test token verification with simulated user token
    token = "dev_alice@enterprise.com"
    verified = verify_token(token)
    print(f"[PASS] Verified token for: {verified['email']} (UID: {verified['uid']})")
    assert verified["email"] == "alice@enterprise.com"
    return verified

def test_multi_user_database():
    print("\n--- [2] Testing Multi-User Database Isolation (Alice vs Bob) ---")
    test_db_path = os.path.join(os.path.dirname(__file__), "data", "test_multiuser.db")
    if os.path.exists(test_db_path):
        try:
            os.remove(test_db_path)
        except Exception:
            pass

    db = Database(test_db_path)

    # 1. Register User A (Alice)
    alice = db.upsert_user("uid_alice_123", email="alice@test.com", display_name="Alice")
    # 2. Register User B (Bob)
    bob = db.upsert_user("uid_bob_456", email="bob@test.com", display_name="Bob")

    # 3. Alice creates chats
    chat_a1 = db.create_chat(alice["uid"], title="Alice Secret Project")
    chat_a2 = db.create_chat(alice["uid"], title="Transformers Research")
    db.save_message(alice["uid"], chat_a1, role="user", content="Classified Alice information.")
    db.save_message(alice["uid"], chat_a1, role="assistant", content="Acknowledged, Alice.")

    # 4. Bob creates chats
    chat_b1 = db.create_chat(bob["uid"], title="Bob Financial Models")
    db.save_message(bob["uid"], chat_b1, role="user", content="Bob confidential revenue figures.")

    # 5. Check Alice's view
    alice_chats = db.get_user_chats(alice["uid"])
    alice_chat_titles = [c["title"] for c in alice_chats]
    print(f"[PASS] Alice chats visible to Alice: {alice_chat_titles}")
    assert "Alice Secret Project" in alice_chat_titles
    assert "Transformers Research" in alice_chat_titles
    assert "Bob Financial Models" not in alice_chat_titles, "SECURITY BREACH: Alice can see Bob's chats!"

    # 6. Check Bob's view
    bob_chats = db.get_user_chats(bob["uid"])
    bob_chat_titles = [c["title"] for c in bob_chats]
    print(f"[PASS] Bob chats visible to Bob: {bob_chat_titles}")
    assert "Bob Financial Models" in bob_chat_titles
    assert "Alice Secret Project" not in bob_chat_titles, "SECURITY BREACH: Bob can see Alice's chats!"

    # 7. Check Pinning and Search
    db.toggle_chat_pin(alice["uid"], chat_a2)
    grouped = db.get_user_chats_grouped(alice["uid"])
    print(f"[PASS] Alice Pinned Chats: {[c['title'] for c in grouped['Pinned']]}")
    assert len(grouped["Pinned"]) == 1
    assert grouped["Pinned"][0]["title"] == "Transformers Research"

    search_res = db.get_user_chats(alice["uid"], search_query="Secret")
    print(f"[PASS] Alice search for 'Secret' returned {len(search_res)} chat(s)")
    assert len(search_res) == 1

    return db

def test_rag_isolation():
    print("\n--- [3] Testing Multi-User RAG & FAISS Vector Store Isolation ---")
    db = Database(os.path.join(os.path.dirname(__file__), "data", "test_rag_iso.db"))

    rag_alice = RAGPipeline(uid="uid_alice_rag", db=db)
    rag_bob = RAGPipeline(uid="uid_bob_rag", db=db)

    # Ingest distinct documents for each user
    alice_text = b"Project Pegasus is Alice's top secret quantum neural network architecture."
    bob_text = b"Project Apollo is Bob's financial trading algorithm for equities."

    res_a = rag_alice.ingest_file(alice_text, "alice_notes.txt")
    res_b = rag_bob.ingest_file(bob_text, "bob_notes.txt")

    print(f"[PASS] Ingested Alice doc: {res_a['filename']}, Bob doc: {res_b['filename']}")

    # Alice queries:
    ctx_a, src_a = rag_alice.retrieve_context("What is Project Pegasus?", top_k=2)
    print(f"[PASS] Alice retrieved: {ctx_a[:60]}... (Sources: {[s['source'] for s in src_a]})")
    assert "Pegasus" in ctx_a
    assert "Apollo" not in ctx_a, "SECURITY BREACH: Alice retrieved Bob's document!"

    # Bob queries:
    ctx_b, src_b = rag_bob.retrieve_context("What is Project Apollo?", top_k=2)
    print(f"[PASS] Bob retrieved: {ctx_b[:60]}... (Sources: {[s['source'] for s in src_b]})")
    assert "Apollo" in ctx_b
    assert "Pegasus" not in ctx_b, "SECURITY BREACH: Bob retrieved Alice's document!"

    # Cleanup test rag
    rag_alice.clear_all_documents()
    rag_bob.clear_all_documents()

def test_export():
    print("\n--- [4] Testing Export Module (Markdown & PDF) ---")
    mock_messages = [
        {"role": "user", "content": "Explain Transformers in two sentences.", "created_at": "2026-11-04 10:00:00"},
        {
            "role": "assistant",
            "content": "Transformers are neural network models that rely on self-attention mechanisms to process sequential tokens. They compute representations in parallel rather than recurrently.",
            "sources": [{"source": "attention_paper.pdf", "page": 3, "score": 0.89, "snippet": "Attention is all you need."}],
            "created_at": "2026-11-04 10:00:02"
        }
    ]

    # Test Markdown
    md = export_to_markdown("Transformer Architecture Discussion", mock_messages)
    print(f"[PASS] Exported Markdown ({len(md)} chars):\n{md[:180]}...")
    assert "# Transformer Architecture Discussion" in md
    assert "attention_paper.pdf" in md

    # Test PDF
    pdf_bytes = export_to_pdf("Transformer Architecture Discussion", mock_messages)
    print(f"[PASS] Exported PDF binary size: {len(pdf_bytes)} bytes")
    assert len(pdf_bytes) > 1000
    assert pdf_bytes.startswith(b"%PDF"), "Generated output is not a valid PDF file!"

if __name__ == "__main__":
    test_auth()
    test_multi_user_database()
    test_rag_isolation()
    test_export()
    print("\n" + "=" * 60)
    print("   [SUCCESS] ALL PHASE 3 MULTI-USER TESTS PASSED!")
    print("=" * 60)


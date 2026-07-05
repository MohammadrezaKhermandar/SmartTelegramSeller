import time


def test_session_roundtrip(memory):
    session = memory.get_or_create_session("chat1", "user1")
    assert session["conversation_stage"] == "new"

    memory.update_session("chat1", requirements={"budget": 1000}, conversation_stage="clarifying")
    session = memory.get_or_create_session("chat1", "user1")
    assert session["requirements"] == {"budget": 1000}
    assert session["conversation_stage"] == "clarifying"


def test_recommendations_positions(memory):
    memory.get_or_create_session("chat1", "user1")
    products = [
        {"product_id": "1", "name": "A"},
        {"product_id": "2", "name": "B"},
        {"product_id": "3", "name": "C"},
    ]
    memory.save_recommendations("chat1", products, ["r1", "r2", "r3"])

    second = memory.get_recommendation_by_position("chat1", 2)
    assert second is not None
    assert second["name"] == "B"
    assert second["_reason"] == "r2"


def test_new_recommendations_archive_old(memory):
    memory.get_or_create_session("chat1", "user1")
    memory.save_recommendations("chat1", [{"product_id": "1", "name": "old"}])
    memory.save_recommendations("chat1", [{"product_id": "2", "name": "new"}])
    active = memory.get_active_recommendations("chat1")
    assert len(active) == 1
    assert active[0]["name"] == "new"


def test_followups_lifecycle(memory):
    memory.get_or_create_session("chat1", "user1")
    memory.schedule_followups("chat1")
    # nothing due yet
    assert memory.due_followups() == []

    # force both followups to be due now
    with memory._lock:
        memory._conn.execute("UPDATE followups SET due_at = ?", (time.time() - 1,))
        memory._conn.commit()
    due = memory.due_followups()
    assert {f["kind"] for f in due} == {"idle_1h", "purchase_2d"}

    for f in due:
        memory.mark_followup_sent(f["id"])
    assert memory.due_followups() == []


def test_purchase_status(memory):
    memory.get_or_create_session("chat1", "user1")
    assert memory.get_purchase_status("chat1") == "browsing"
    memory.update_session("chat1", purchase_status="purchased")
    assert memory.get_purchase_status("chat1") == "purchased"


def test_reset_chat_clears_all_state(memory):
    memory.get_or_create_session("chat1", "user1")
    memory.update_session(
        "chat1",
        requirements={"budget": 50_000_000},
        conversation_stage="recommended",
    )
    memory.save_recommendations("chat1", [{"product_id": "1", "name": "A"}])
    memory.schedule_followups("chat1")

    memory.reset_chat("chat1")

    session = memory.get_or_create_session("chat1", "user1")
    assert session["conversation_stage"] == "new"
    assert session["requirements"] == {}
    assert memory.get_active_recommendations("chat1") == []
    assert memory.due_followups() == []

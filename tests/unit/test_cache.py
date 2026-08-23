from pathlib import Path

from sentiment_agent.llm.cache import SQLiteResponseCache, build_cache_key


def test_cache_key_is_independent_of_mapping_order() -> None:
    first = build_cache_key({"model": "x", "temperature": 0, "messages": [{"role": "user", "content": "hi"}]})
    second = build_cache_key({"messages": [{"content": "hi", "role": "user"}], "temperature": 0, "model": "x"})

    assert first == second


def test_sqlite_cache_round_trip(tmp_path: Path) -> None:
    cache = SQLiteResponseCache(tmp_path / "responses.sqlite3")
    payload = {"choices": [{"message": {"content": "{}"}}]}

    assert cache.get("missing") is None
    cache.put("key", payload)

    assert cache.get("key") == payload

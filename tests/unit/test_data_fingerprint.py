from pathlib import Path

from sentiment_agent.data.fingerprint import fingerprint_file


def test_fingerprint_is_stable_and_content_sensitive(tmp_path: Path) -> None:
    path = tmp_path / "data.json"
    path.write_text("[]", encoding="utf-8")
    first = fingerprint_file(path)
    assert fingerprint_file(path) == first
    path.write_text("[1]", encoding="utf-8")
    assert fingerprint_file(path) != first

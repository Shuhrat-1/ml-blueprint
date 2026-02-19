from pathlib import Path

from mlb.core.logging import setup_logger


def test_setup_logger_writes_jsonl(tmp_path: Path) -> None:
    log_file = tmp_path / "logs.jsonl"
    logger = setup_logger(name="mlb_test_logger", log_file=log_file)
    logger.info("hello")

    assert log_file.exists()
    content = log_file.read_text(encoding="utf-8").strip()
    assert "hello" in content
    assert content.startswith("{") and content.endswith("}")

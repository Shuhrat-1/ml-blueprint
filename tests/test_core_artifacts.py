from mlb.core.artifacts import create_run_dir, save_json, save_text, save_yaml
from mlb.core.paths import Paths


def test_create_run_dir(tmp_path) -> None:
    paths = Paths(root=tmp_path, data_dir=tmp_path / "data", artifacts_dir=tmp_path / "artifacts")
    run = create_run_dir(paths=paths, name="test_run", prefix="unit", with_timestamp=False)

    assert run.run_dir.exists()
    assert run.run_dir.name == "unit_test_run"
    assert run.config_path.name == "config_resolved.yaml"
    assert run.metrics_path.name == "metrics.json"
    assert run.logs_path.name == "logs.jsonl"


def test_save_helpers(tmp_path) -> None:
    p_json = tmp_path / "a" / "x.json"
    p_txt = tmp_path / "b" / "x.txt"
    p_yaml = tmp_path / "c" / "x.yaml"

    save_json(p_json, {"a": 1})
    save_text(p_txt, "hello")
    save_yaml(p_yaml, {"b": 2})

    assert p_json.exists() and p_json.read_text(encoding="utf-8")
    assert p_txt.read_text(encoding="utf-8") == "hello"
    assert p_yaml.exists() and p_yaml.read_text(encoding="utf-8")

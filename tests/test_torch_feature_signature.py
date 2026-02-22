from mlb.models.torch.dataset import CatVocab
from mlb.models.torch.signature import compute_feature_signature


def test_feature_signature_stable() -> None:
    schema = {
        "schema_version": 1,
        "target": "y",
        "features": {"numeric": ["a"], "categorical": ["c"], "text": [], "datetime": []},
        "feature_order": ["a", "c"],
    }
    vocabs = {"c": CatVocab(mapping={"X": 2, "Y": 3})}
    num_stats = {"mean": [0.0], "std": [1.0]}

    s1 = compute_feature_signature(schema_resolved=schema, vocabs=vocabs, num_stats=num_stats)
    s2 = compute_feature_signature(schema_resolved=schema, vocabs=vocabs, num_stats=num_stats)
    assert s1 == s2


def test_feature_signature_changes_if_vocab_size_changes() -> None:
    schema = {
        "schema_version": 1,
        "target": "y",
        "features": {"numeric": ["a"], "categorical": ["c"], "text": [], "datetime": []},
        "feature_order": ["a", "c"],
    }
    vocabs1 = {"c": CatVocab(mapping={"X": 2})}
    vocabs2 = {"c": CatVocab(mapping={"X": 2, "Y": 3})}
    num_stats = {"mean": [0.0], "std": [1.0]}

    s1 = compute_feature_signature(schema_resolved=schema, vocabs=vocabs1, num_stats=num_stats)
    s2 = compute_feature_signature(schema_resolved=schema, vocabs=vocabs2, num_stats=num_stats)
    assert s1 != s2
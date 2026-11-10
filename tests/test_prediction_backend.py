import pandas as pd
import pytest
from prediction_backend import prepare_input


def test_prepare_input_orders_features_and_drops_sample_id():
    frame = pd.DataFrame({"sample_id": ["S1"], "gene_b": [2], "gene_a": [1]})
    result = prepare_input(frame, ["gene_a", "gene_b"])
    assert list(result.columns) == ["gene_a", "gene_b"]
    assert result.iloc[0].tolist() == [1, 2]


def test_prepare_input_rejects_missing_features():
    with pytest.raises(ValueError, match="Missing"):
        prepare_input(pd.DataFrame({"gene_a": [1]}), ["gene_a", "gene_b"])

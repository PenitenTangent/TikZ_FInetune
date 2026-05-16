from tikz_mlx.example_index import assign_example_index


def test_assign_example_index_sets_top_level_and_metadata() -> None:
    record = {"sample_id": "a", "metadata": {"source": "unit"}}

    assign_example_index(record, 7)

    assert record["example_index"] == 7
    assert record["metadata"]["example_index"] == 7
    assert record["metadata"]["source"] == "unit"


def test_assign_example_index_creates_metadata_when_missing() -> None:
    record = {"sample_id": "a"}

    assign_example_index(record, 0)

    assert record["example_index"] == 0
    assert record["metadata"] == {"example_index": 0}

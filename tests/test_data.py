from src.data import FEATURES, TARGET, load_raw


def test_raw_data_has_expected_shape_and_columns():
    # The first run downloads from UCI and later runs read the cached CSV.
    df = load_raw()
    assert df.shape == (45730, 10)  # 9 features + 1 target
    assert list(df.columns) == FEATURES + [TARGET]

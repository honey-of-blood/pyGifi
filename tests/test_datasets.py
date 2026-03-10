import pytest
import pandas as pd
from pygifi.datasets import get_dataset


def test_get_dataset_success():
    df = get_dataset('hartigan')
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (24, 6)


def test_get_dataset_invalid():
    with pytest.raises(FileNotFoundError, match="Dataset 'invalid_dataset' not found"):
        get_dataset('invalid_dataset')


def test_get_dataset_all_bundled():
    # Not all might be generated yet, but let's test a few common ones if they
    # exist
    for name in ['hartigan', 'galo']:
        df = get_dataset(name)
        assert not df.empty
        assert isinstance(df, pd.DataFrame)

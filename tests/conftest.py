"""Shared fixtures for dp_demo tests."""

import pytest
from dataset import CensusDataset, generate_dataset


@pytest.fixture(scope="session")
def records():
    """1000-record deterministic census sample (shared across all tests)."""
    return generate_dataset(1000)


@pytest.fixture(scope="session")
def ds(records):
    return CensusDataset(records)


@pytest.fixture(scope="session")
def summary(ds):
    return ds.summary()

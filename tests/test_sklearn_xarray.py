#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for `sklearn_xarray` package."""

import pytest

import numpy as np
import xarray as xr
import pandas as pd

from sklearn_xarray import XarrayUnion, Select, Stacker, concat_multi_indexes
from sklearn.pipeline import make_pipeline


@pytest.fixture
def response():
    """Sample pytest fixture.

    See more at: http://doc.pytest.org/en/latest/fixture.html
    """
    # import requests
    # return requests.get('https://github.com/audreyr/cookiecutter-pypackage')


@pytest.fixture
def synthetic_data():

    # Make synthetic data
    lat, lon = np.ogrid[-45:45:50j, 0:360:100j]
    noise = np.random.randn(lat.shape[0], lon.shape[1])

    data_vars = {
        'a': (['lat', 'lon'], np.sin(lat / 90 + lon / 100)),
        'b': (['lat', 'lon'], np.cos(lat / 90 + lon / 100)),
        'noise': (['lat', 'lon'], noise)
    }

    coords = {'lat': lat.ravel(), 'lon': lon.ravel()}
    dataset = xr.Dataset(data_vars, coords)

    return dataset


def test_content(response):
    """Sample pytest test function with the pytest fixture as an argument."""
    # from bs4 import BeautifulSoup
    # assert 'GitHub' in BeautifulSoup(response.content).title.string


def test_xarray_union(synthetic_data):
    """Test xarray_union"""
    union = XarrayUnion(
        [
            ('a', make_pipeline(Select('a'))),
            ('b', make_pipeline(Select('b'))),
        ],
        feature_dims=['lat'])

    xformed = union.fit_transform(synthetic_data)
    assert tuple(xformed.indexes['features']) == ('a', 'b')

    ixform = union.inverse_transform(xformed)
    assert ixform == synthetic_data


def test_concat_multindexes():
    tuples = [('a', 0), ('a', 1)]
    idx = pd.MultiIndex.from_tuples(tuples, names=[0, 1])
    cat = concat_multi_indexes((idx, idx))
    idx = pd.MultiIndex.from_tuples(tuples + tuples)

    assert all(cat == idx)

    idx1 = pd.MultiIndex.from_tuples(tuples, names=['n1', 'n2'])
    idx2 = pd.MultiIndex.from_tuples(tuples, names=['m1', 'm2'])

    tuples = [
        ('a', 0, None, None),
        ('a', 1, None, None),
        (None, None, 'a', 0),
        (None, None, 'a', 1)
    ]

    expected = pd.MultiIndex.from_tuples(tuples, names=idx1.names + idx2.names)
    cat = concat_multi_indexes((idx1, idx2))
    assert str(cat) == str(expected)

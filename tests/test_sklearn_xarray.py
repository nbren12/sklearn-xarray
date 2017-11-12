#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `sklearn_xarray` package."""

import pytest

import numpy as np
import xarray as xr

from sklearn_xarray import XarrayUnion, Select, Stacker
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
        'a': (['lat', 'lon'], np.sin(lat/90 + lon/100)),
        'b': (['lat', 'lon'], np.cos(lat/90 + lon/100)),
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
            ('a', make_pipeline(Select('a'), Stacker())),
            ('b', make_pipeline(Select('b'), Stacker())),
        ]
    )

    xformed = union.fit_transform(synthetic_data)
    assert tuple(xformed.indexes['features']) == ('a', 'b')

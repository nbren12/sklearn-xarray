#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

with open('README.md') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [
    'sphinx-gallery',
    'xarray'
    # TODO: put package requirements here
]

setup_requirements = [
    'pytest-runner',
    # TODO(nbren12): put setup requirements (distutils extensions, etc.) here
]

test_requirements = [
    'pytest',
    # TODO: put package test requirements here
]

setup(
    name='sklearn_xarray',
    version='0.0.0',
    description="Munge xarray objects into sklearn pipelines",
    long_description=readme + '\n\n' + history,
    author="Noah D Brenowitz",
    author_email='nbren12@gmail.com',
    url='https://github.com/nbren12/sklearn_xarray',
    packages=find_packages(include=['sklearn_xarray']),
    entry_points={
        'console_scripts': [
            'sklearn_xarray=sklearn_xarray.cli:main'
        ]
    },
    include_package_data=True,
    install_requires=requirements,
    license="MIT license",
    zip_safe=False,
    keywords='sklearn_xarray',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        "Programming Language :: Python :: 2",
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],
    test_suite='tests',
    tests_require=test_requirements,
    setup_requires=setup_requirements,
)

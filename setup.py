#!/usr/bin/env python
# coding=utf-8
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="Constrained_GaussianProcess",
    version="0.0.5",
    author="Liaowang Huang",
    author_email="liahuang@student.ethz.ch",
    description="Implementation of Python package for Fitting and Inference of Linearly Constrained Gaussian Processes",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/liaowangh/constrained_gp",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)

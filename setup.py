#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name='eft',
    version='0.0.0',
    description='Enformer fine tuning (EFT) for DNA diffusion expression oracle.',
    author='',
    author_email='',
    # REPLACE WITH YOUR OWN GITHUB PROJECT LINK
    url='https://github.com/wconnell/enformer-finetune',
    install_requires=['pytorch-lightning'],
    packages=find_packages(),
)


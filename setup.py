import os
from setuptools import setup, find_packages


setup(
    name="bnb",
    author="Ruben van de Geer",
    packages=find_packages(exclude=['data', 'figures', 'output', 'notebooks']),
)
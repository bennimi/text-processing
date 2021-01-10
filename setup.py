# -*- coding: utf-8 -*-
"""
Text Processing Pipeline

@author: bennimi
"""

from setuptools import setup

requirements = ["num2words >= 0.5.10","gensim >= 3.8.3",
                "spacy >= 2.3.5" ] 



setup(
    name='Git_Repo_Nudge',
    version='0.0.3',
    description='My private package from private github repo',
    url='https://github.com/bennimi/Git_Repo_Nudge.git',
    author='Benedikt Mueller',
    author_email='benedikt.mueller.2019@uni.strath.ac.uk',
    license=' SIL Open Font License 1.1',
    packages=['Git_Repo_Nudge'],
    install_requires=requirements,
    zip_safe=False
)

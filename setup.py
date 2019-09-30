#nsml: w4jinsuk/pytorch-audio:latest
from distutils.core import setup
import setuptools

setup(
    name='speech_hackathon',
    version='1.0',
    install_requires=[
        'python-Levenshtein',
        'torchaudio',
        'librosa'
    ]
)

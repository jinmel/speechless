FROM nvcr.io/nvidia/pytorch:19.09-py3

RUN pip install torchaudio python-Levenshtein
RUN pip install librosa

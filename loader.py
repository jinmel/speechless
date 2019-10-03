"""
Copyright 2019-present NAVER Corp.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

#-*- coding: utf-8 -*-

import concurrent
import os
import sys
import math
import wavio
import time
import torch
import random
import threading
import librosa
import logging
import numpy as np
from torch.utils.data import Dataset
from joblib import Memory
import tempfile
import shutil

logger = logging.getLogger('root')
FORMAT = "[%(asctime)s %(filename)s:%(lineno)s - %(funcName)s()] %(message)s"
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format=FORMAT)
logger.setLevel(logging.DEBUG)

CACHE_DIR = './cache'
memory = Memory(CACHE_DIR, verbose=0)

PAD = 0
N_FFT = 320
N_MFCC = 40
N_MELS = 80
SAMPLE_RATE = 16000

def clear_joblib_cache():
    logger.info('Clear Joblib cache at: %s' % CACHE_DIR)
    shutil.rmtree(CACHE_DIR)

def augment_audio_with_sox(path, sample_rate, tempo, gain):
    """
    Changes tempo and gain of the recording with sox and loads it.
    """
    with tempfile.NamedTemporaryFile(suffix=".wav") as augmented_file:
        augmented_filename = augmented_file.name
        sox_augment_params = ["tempo", "{:.3f}".format(tempo), "gain", "{:.3f}".format(gain)]
        sox_params = "sox \"{}\" -r {} -c 1 -b 16 -e si {} {} >/dev/null 2>&1".format(path, sample_rate,
                                                                                      augmented_filename,
                                                                                      " ".join(sox_augment_params))
        os.system(sox_params)
        y, _ = librosa.load(augmented_filename)
        return y

def load_randomly_augmented_audio(path, sample_rate=SAMPLE_RATE,
                                  tempo_range=(0.85, 1.15), gain_range=(-6, 8)):
    """
    Picks tempo and gain uniformly, applies it to the utterance by using sox utility.
    Returns the augmented utterance.
    """
    low_tempo, high_tempo = tempo_range
    tempo_value = np.random.uniform(low=low_tempo, high=high_tempo)
    low_gain, high_gain = gain_range
    gain_value = np.random.uniform(low=low_gain, high=high_gain)
    audio = augment_audio_with_sox(path=path, sample_rate=sample_rate,
                                   tempo=tempo_value, gain=gain_value)
    return audio

def load_audio(path, augment=False):
    if augment:
        return load_randomly_augmented_audio(path), SAMPLE_RATE
    return librosa.load(path)

def load_targets(path):
    target_dict = dict()
    with open(path, 'r') as f:
        for no, line in enumerate(f):
            key, target = line.strip().split(',')
            target_dict[key] = target
    return target_dict

def get_mfcc_feature(filepath, augment=False):
    y, sr = load_audio(filepath, augment)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta_delta = librosa.feature.delta(mfcc, order=2)
    result = np.concatenate((mfcc, mfcc_delta, mfcc_delta_delta), axis=0).T
    return torch.FloatTensor(result)

def get_mel_spectrogram_feature(filepath, augment=False):
    y, sr = load_audio(filepath, augment)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS,
                                              n_fft=N_FFT,
                                              win_length=int(N_FFT),
                                              hop_length=int(0.01*SAMPLE_RATE),
                                              center=False,
                                              window='hamming')
    return torch.FloatTensor(mel_spec.T)

def get_spectrogram_feature(filepath, augment=False):
    y, sr = load_audio(filepath, augment)
    D = librosa.stft(y, n_fft=N_FFT, hop_length=int(0.01*SAMPLE_RATE),
                     win_length=int(N_FFT), window='hamming')
    spect, phase = librosa.magphase(D)
    spect = np.log1p(spect)
    spect = torch.FloatTensor(spect)
    return spect.T

def normalize_feature(feature):
    mean = feature.mean()
    std = feature.std()
    feature.add_(-mean)
    feature.div_(std)
    return feature

def get_script(script, bos_id, eos_id):
    tokens = script.split(' ')
    result = list()
    result.append(bos_id)
    for i in range(len(tokens)):
        if len(tokens[i]) > 0:
            result.append(int(tokens[i]))
    result.append(eos_id)
    return result

class BaseDataset(Dataset):
    def __init__(self, wav_paths, script_paths, target_dict, feature='spec',
                 bos_id=1307, eos_id=1308, normalize=True):
        self.wav_paths = wav_paths
        self.script_paths = script_paths
        self.target_dict = target_dict
        self.bos_id, self.eos_id = bos_id, eos_id
        self.feature = feature
        self.normalize = normalize

    def __len__(self):
        return len(self.wav_paths)

    def __getitem__(self, index):
        return self.getitem(index)

    def count(self):
        return len(self.wav_paths)

    def get_feature_func(self):
        if self.feature == 'mfcc':
            get_feature = get_mfcc_feature
        elif self.feature == 'melspec':
            get_feature = get_mel_spectrogram_feature
        elif self.feature == 'spec':
            get_feature = get_spectrogram_feature
        else:
            raise ValueError('Unsupported feature: %s' % self.feature)

        return get_feature

    def getitem(self, idx):
        get_feature = self.get_feature_func()
        get_feature_cache = memory.cache(get_feature)
        get_script_cache = memory.cache(get_script)
        feat = get_feature_cache(self.wav_paths[idx], False)
        if self.normalize:
            feat = normalize_feature(feat)
        key = self.script_paths[idx].split('/')[-1].split('.')[0]
        script = get_script_cache(self.target_dict[key],
                                  self.bos_id, self.eos_id)
        return feat, script

class SpecaugDataset(BaseDataset):
    def getitem(self, idx):
        get_feature = self.get_feature_func()
        get_script_cache = memory.cache(get_script)
        feat = get_feature(self.wav_paths[idx], True)
        if self.normalize:
            feat = normalize_feature(feat)
        key = self.script_paths[idx].split('/')[-1].split('.')[0]
        script = get_script_cache(self.target_dict[key],
                                  self.bos_id, self.eos_id)
        return feat, script

def collate_fn(batch):
    def seq_length_(p):
        return len(p[0])

    def target_length_(p):
        return len(p[1])

    seq_lengths = [len(s[0]) for s in batch]
    target_lengths = [len(s[1]) for s in batch]

    max_seq_sample = max(batch, key=seq_length_)[0]
    max_target_sample = max(batch, key=target_length_)[1]

    max_seq_size = max_seq_sample.size(0)
    max_target_size = len(max_target_sample)

    feat_size = max_seq_sample.size(1)
    batch_size = len(batch)

    seqs = torch.zeros(batch_size, max_seq_size, feat_size)

    targets = torch.zeros(batch_size, max_target_size).to(torch.long)
    targets.fill_(PAD)

    for x in range(batch_size):
        sample = batch[x]
        tensor = sample[0]
        target = sample[1]
        seq_length = tensor.size(0)
        seqs[x].narrow(0, 0, seq_length).copy_(tensor)
        targets[x].narrow(0, 0, len(target)).copy_(torch.LongTensor(target))

    return seqs, targets, seq_lengths, target_lengths

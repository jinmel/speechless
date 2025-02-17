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

import os
import sys
import time
import math
import wavio
import argparse
import queue
import shutil
import random
import math
import time
import torch
import logging
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.optim as optim
import Levenshtein as Lev

import label_loader
from loader import *
from models import EncoderRNN, DecoderRNN, Seq2seq

import nsml
from nsml import GPU_NUM, DATASET_PATH, DATASET_NAME, HAS_DATASET

char2index = dict()
index2char = dict()
SOS_token = 0
EOS_token = 0
PAD_token = 0

if HAS_DATASET == False:
    DATASET_PATH = './sample_dataset'

DATASET_PATH = os.path.join(DATASET_PATH, 'train')

def label_to_string(labels):
    if len(labels.shape) == 1:
        sent = str()
        for i in labels:
            if i.item() == EOS_token:
                break
            sent += index2char[i.item()]
        return sent

    elif len(labels.shape) == 2:
        sents = list()
        for i in labels:
            sent = str()
            for j in i:
                if j.item() == EOS_token:
                    break
                sent += index2char[j.item()]
            sents.append(sent)

        return sents

def char_distance(ref, hyp):
    ref = ref.replace(' ', '')
    hyp = hyp.replace(' ', '')

    dist = Lev.distance(hyp, ref)
    length = len(ref.replace(' ', ''))

    return dist, length

def get_distance(ref_labels, hyp_labels, display=False):
    total_dist = 0
    total_length = 0
    for i in range(len(ref_labels)):
        ref = label_to_string(ref_labels[i])
        hyp = label_to_string(hyp_labels[i])
        dist, length = char_distance(ref, hyp)
        total_dist += dist
        total_length += length
        if display:
            cer = total_dist / total_length
            logger.debug('%d (%0.4f)\n(%s)\n(%s)' % (i, cer, ref, hyp))
    return total_dist, total_length


def train(model, data_loader, criterion, optimizer, device,
          train_begin, print_batch=5, teacher_forcing_ratio=1):
    total_loss = 0.
    total_num = 0
    total_dist = 0
    total_length = 0
    total_sent_num = 0
    batch = 0

    model.train()

    total_batch_size = len(data_loader)
    logger.info('train() start')

    begin = epoch_begin = time.time()

    for batch_index, batch in enumerate(data_loader):
        feats, scripts, feat_lengths, script_lengths = batch
        optimizer.zero_grad()

        feats = feats.to(device)
        scripts = scripts.to(device)

        src_len = scripts.size(1)
        target = scripts[:, 1:]

        model.module.flatten_parameters()
        logit = model(feats, feat_lengths, scripts, teacher_forcing_ratio=teacher_forcing_ratio)

        logit = torch.stack(logit, dim=1).to(device)

        y_hat = logit.max(-1)[1]

        loss = criterion(logit.contiguous().view(-1, logit.size(-1)), target.contiguous().view(-1))
        total_loss += loss.item()
        total_num += sum(feat_lengths)

        display = random.randrange(0, 100) == 0
        dist, length = get_distance(target, y_hat, display=display)
        total_dist += dist
        total_length += length

        total_sent_num += target.size(0)

        loss.backward()
        optimizer.step()

        if batch_index % print_batch == 0:
            current = time.time()
            elapsed = current - begin
            epoch_elapsed = (current - epoch_begin) / 60.0
            train_elapsed = (current - train_begin) / 3600.0

            logger.info('batch: {:4d}/{:4d}, loss: {:.4f}, cer: {:.2f}, elapsed: {:.2f}s {:.2f}m {:.2f}h'
                .format(batch_index,
                        total_batch_size,
                        total_loss / total_num,
                        total_dist / total_length,
                        elapsed, epoch_elapsed, train_elapsed))
            begin = time.time()

            nsml.report(False,
                        step=train.cumulative_batch_count,
                        train_step__loss=total_loss/total_num,
                        train_step__cer=total_dist/total_length)

        train.cumulative_batch_count += 1

    logger.info('train() completed')
    return total_loss / total_num, total_dist / total_length


train.cumulative_batch_count = 0


def evaluate(model, dataloader, criterion, device):
    logger.info('evaluate() start')
    total_loss = 0.
    total_num = 0
    total_dist = 0
    total_length = 0
    total_sent_num = 0

    model.eval()

    with torch.no_grad():
        for batch_index, batch in enumerate(dataloader):
            feats, scripts, feat_lengths, script_lengths = batch
            feats = feats.to(device)
            scripts = scripts.to(device)

            src_len = scripts.size(1)
            target = scripts[:, 1:]

            model.module.flatten_parameters()
            logit = model(feats, feat_lengths, scripts, teacher_forcing_ratio=0.0)

            logit = torch.stack(logit, dim=1).to(device)
            y_hat = logit.max(-1)[1]

            loss = criterion(logit.contiguous().view(-1, logit.size(-1)), target.contiguous().view(-1))
            total_loss += loss.item()
            total_num += sum(feat_lengths)

            display = random.randrange(0, 100) == 0
            dist, length = get_distance(target, y_hat, display=True)
            total_dist += dist
            total_length += length
            total_sent_num += target.size(0)

    logger.info('evaluate() completed')
    return total_loss / total_num, total_dist / total_length

def save_model(model, optimizer, path):
    state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    torch.save(state, path)

def bind_model(model, optimizer=None, feature='spec'):
    def load(filename, **kwargs):
        state = torch.load(os.path.join(filename, 'model.pt'))
        model.load_state_dict(state['model'])
        if 'optimizer' in state and optimizer:
            optimizer.load_state_dict(state['optimizer'])
        print('Model loaded')

    def save(filename, **kwargs):
        state = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        torch.save(state, os.path.join(filename, 'model.pt'))

    def infer(wav_path):
        model.eval()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if feature == 'spec':
            input = get_spectrogram_feature(wav_path, False).unsqueeze(0)
        elif feature == 'mfcc':
            input = get_mfcc_feature(wav_path, False).unsqueeze(0)
        elif feature == 'melspec':
            input = get_mel_spectrogram_feature(wav_path, False).unsqueeze(0)
        else:
            raise ValueError('invalid feature %s' % feature)

        input = normalize_feature(input)
        input = input.to(device)

        logit = model(input_variable=input, input_lengths=None, teacher_forcing_ratio=0)
        logit = torch.stack(logit, dim=1).to(device)

        y_hat = logit.max(-1)[1]
        hyp = label_to_string(y_hat)
        return hyp[0]

    nsml.bind(save=save, load=load, infer=infer) # 'nsml.bind' function must be called at the end.

def split_dataset(config, wav_paths, script_paths, target_dict, feature, valid_ratio=0.05):
    records_num = len(wav_paths)
    batch_num = math.ceil(records_num / config.batch_size)

    valid_batch_num = math.ceil(batch_num * valid_ratio)
    train_batch_num = batch_num - valid_batch_num

    split_index = train_batch_num * config.batch_size

    train_dataset = SpecaugDataset(wav_paths[:split_index],
                                   script_paths[:split_index],
                                   target_dict, feature, SOS_token, EOS_token)
    valid_dataset = BaseDataset(wav_paths[split_index:],
                                script_paths[split_index:],
                                target_dict, feature, SOS_token, EOS_token)

    return train_dataset, valid_dataset

def main():

    global char2index
    global index2char
    global SOS_token
    global EOS_token
    global PAD_token

    parser = argparse.ArgumentParser(description='Speech hackathon Baseline')
    parser.add_argument('--hidden_size', type=int, default=512, help='hidden size of model (default: 256)')
    parser.add_argument('--layer_size', type=int, default=3, help='number of layers of model (default: 3)')
    parser.add_argument('--dropout', type=float, default=0.2, help='dropout rate in training (default: 0.2)')
    parser.add_argument('--bidirectional', action='store_true', help='use bidirectional RNN for encoder (default: False)')
    parser.add_argument('--use_attention', action='store_true', help='use attention between encoder-decoder (default: False)')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size in training (default: 32)')
    parser.add_argument('--workers', type=int, default=4, help='number of workers in dataset loader (default: 4)')
    parser.add_argument('--max_epochs', type=int, default=10, help='number of max epochs in training (default: 10)')
    parser.add_argument('--lr', type=float, default=1e-04, help='learning rate (default: 0.0001)')
    parser.add_argument('--teacher_forcing', type=float, default=0.5, help='teacher forcing ratio in decoder (default: 0.5)')
    parser.add_argument('--max_len', type=int, default=80, help='maximum characters of sentence (default: 80)')
    parser.add_argument('--no_cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--save_name', type=str, default='model', help='the name of model in nsml or local')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument("--pause", type=int, default=0)
    parser.add_argument('--rnn_cell', type=str, default='gru')
    parser.add_argument("--iteration", type=int, default=0)
    parser.add_argument('--feature', type=str, default='spec')
    parser.add_argument('--save_dir', type=str, default='')

    args = parser.parse_args()

    char2index, index2char = label_loader.load_label('./hackathon.labels')
    SOS_token = char2index['<s>']
    EOS_token = char2index['</s>']
    PAD_token = char2index['_']

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if args.cuda else 'cpu')

    logger.info('Using %s as feature' % args.feature)
    if args.save_dir:
        logger.info('Save directory: %s' % args.save_dir)
        os.makedirs(args.save_dir, exist_ok=True)

    # N_FFT: defined in loader.py
    if args.feature == 'mfcc':
        feature_size = N_MFCC * 3 # concat of mfcc, mfcc' mfcc''
    elif args.feature == 'melspec':
        feature_size = N_MELS
    elif args.feature == 'spec':
        feature_size = N_FFT / 2 + 1
    else:
        raise ValueError('Unsupported feature %s' % args.feature)

    enc = EncoderRNN(feature_size, args.hidden_size,
                     input_dropout_p=args.dropout, dropout_p=args.dropout,
                     n_layers=args.layer_size, bidirectional=args.bidirectional, rnn_cell=args.rnn_cell, variable_lengths=False)

    dec = DecoderRNN(len(char2index), args.max_len, args.hidden_size * (2 if args.bidirectional else 1),
                     SOS_token, EOS_token,
                     n_layers=args.layer_size, rnn_cell=args.rnn_cell, bidirectional=args.bidirectional,
                     input_dropout_p=args.dropout, dropout_p=args.dropout, use_attention=args.use_attention)

    model = Seq2seq(enc, dec)
    model.flatten_parameters()

    for param in model.parameters():
        param.data.uniform_(-0.08, 0.08)

    model = nn.DataParallel(model).to(device)

    optimizer = optim.Adam(model.module.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(reduction='sum', ignore_index=PAD_token).to(device)

    bind_model(model, optimizer, args.feature)
    if args.pause != 1:
        nsml.load(checkpoint='10', session='team236/sr-hack-2019-dataset/122')
        nsml.save('init')
        logger.info('Saved!')

    if args.pause == 1:
        nsml.paused(scope=locals())

    if args.mode != "train":
        return

    data_list = os.path.join(DATASET_PATH, 'train_data', 'data_list.csv')
    wav_paths = list()
    script_paths = list()
    # load all target scripts for reducing disk i/o
    target_path = os.path.join(DATASET_PATH, 'train_label')
    target_dict = load_targets(target_path)

    with open(data_list, 'r') as f:
        for line in f:
            # line: "aaa.wav,aaa.label"
            wav_path, script_path = line.strip().split(',')
            wav_paths.append(os.path.join(DATASET_PATH, 'train_data', wav_path))
            script_paths.append(os.path.join(DATASET_PATH, 'train_data', script_path))

    best_loss = 1e10
    begin_epoch = 0

    train_dataset, valid_dataset = split_dataset(
        args, wav_paths, script_paths, target_dict, args.feature, valid_ratio=0.05)

    train_begin = time.time()

    for epoch in range(begin_epoch, args.max_epochs):
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, collate_fn=collate_fn)

        train_loss, train_cer = train(
            model, train_loader, criterion, optimizer, device,
            train_begin, 10, args.teacher_forcing)

        logger.info('Epoch %d (Training) Loss %0.4f CER %0.4f' % (epoch, train_loss, train_cer))

        valid_loader = torch.utils.data.DataLoader(
            valid_dataset, batch_size=4, shuffle=False,
            num_workers=args.workers, collate_fn=collate_fn)

        eval_loss, eval_cer = evaluate(model, valid_loader, criterion, device)
        logger.info('Epoch %d (Evaluate) Loss %0.4f CER %0.4f' % (epoch, eval_loss, eval_cer))

        nsml.report(False,
                    step=epoch, train_epoch__loss=train_loss,
                    train_epoch__cer=train_cer,
                    eval__loss=eval_loss, eval__cer=eval_cer)

        best_model = (eval_loss < best_loss)
        nsml.save(args.save_name)
        nsml.save(str(epoch))

        if args.save_dir:
            save_model(model, optimizer,
                       os.path.join(args.save_dir,
                                    './epoch-%d-cer-%d.pt' % (epoch, eval_cer)))

        if best_model:
            nsml.save('best')
            best_loss = eval_loss


if __name__ == "__main__":
    main()

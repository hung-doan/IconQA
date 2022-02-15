from __future__ import print_function
import os
import sys
import json
import _pickle as cPickle # python3
import numpy as np

import torch
from torch.utils.data import Dataset

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools import utils

from transformers import AutoTokenizer
bert_models = {'bert-base-uncased': 'bert-base-uncased',
               'bert-tiny':   'google/bert_uncased_L-2_H-128_A-2',
               'bert-mini':   'google/bert_uncased_L-4_H-256_A-4',
               'bert-small':  'google/bert_uncased_L-4_H-512_A-8',
               'bert-medium': 'google/bert_uncased_L-8_H-512_A-8',
               'bert-base':   'google/bert_uncased_L-12_H-768_A-12'}


class Dictionary(object):
    def __init__(self, word2idx=None, idx2word=None):
        if word2idx is None:
            word2idx = {}
        if idx2word is None:
            idx2word = []
        self.word2idx = word2idx
        self.idx2word = idx2word

    @property
    def ntoken(self):
        return len(self.word2idx)

    @property
    def padding_idx(self):
        return len(self.word2idx)

    def tokenize(self, sentence, add_word):
        sentence = sentence.lower()
        sentence = sentence.replace(',', '').replace('?', '').replace('\'s', ' \'s')
        words = sentence.split()
        tokens = []
        if add_word:
            for w in words:
                tokens.append(self.add_word(w))
        else:
            for w in words:
                tokens.append(self.word2idx[w])
        return tokens

    def dump_to_file(self, path):
        cPickle.dump([self.word2idx, self.idx2word], open(path, 'wb'))
        print('dictionary dumped to %s' % path)

    @classmethod
    def load_from_file(cls, path):
        print('\nloading dictionary from %s' % path)
        word2idx, idx2word = cPickle.load(open(path, 'rb'))
        d = cls(word2idx, idx2word) # initialize the instance
        print('vocabulary number in the dictionary:', len(idx2word))
        return d

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


def _simplify_question(ques):
    """
    Simplify question: remove verbose sentences in the question.
    """
    sentences = ques.split(". ")
    if len(sentences) > 1 and "Count" in sentences[0] and " by " in sentences[0]:
        ques = ". ".join(sentences[1:])
        return ques
    else:
        return ques


def _load_dataset(dataroot, name, task):
    """
    Load the IconQA dataset.
    - dataroot: root path of dataset
    - name: 'train', 'val', 'test', 'traninval', 'minitrain', 'minival', 'minitest'
    - task: 'choose_img'
    """
    problems =  json.load(open(os.path.join(dataroot, 'iconqa_data', 'problems.json')))
    pid_splits = json.load(open(os.path.join(dataroot, 'iconqa_data', 'pid_splits.json')))
    
    pids = pid_splits['%s_%s' % (task, name)]
    print("problem number for %s_%s:" % (task, name), len(pids))

    entries = []
    for pid in pids:
        prob = {}
        prob['question_id'] = pid
        prob['image_id'] = pid
        prob['question'] = _simplify_question(problems[pid]['question'])
        prob['ques_type'] = problems[pid]['ques_type']

        utils.assert_eq(task, prob['ques_type'])

        # answer to label
        if 'test' not in name: # train, val
            ans = str(problems[pid]['answer'])
            prob['answer'] = ans
            prob['answer_label'] = int(ans)
        else: # test
            ans = str(problems[pid]['answer'])
            prob['answer'] = ans
            prob['answer_label'] = int(ans)

        if task == 'choose_img':
            prob['choices'] = problems[pid]['choices']

        entries.append(prob)

    return entries


class IconQAFeatureDataset(Dataset):
    def __init__(self, name, task, feat_label, choice_feat, dataroot, dictionary, lang_model, max_length, num_patches):
        super(IconQAFeatureDataset, self).__init__ ()
        assert name in ['train', 'val', 'test', 'traninval', 'minitrain', 'minival', 'minitest']
        assert task in ['fill_in_blank', 'choose_txt', 'choose_img']
        assert 'bert' in lang_model

        self.task = task
        self.dictionary = dictionary
        self.lang_model = lang_model
        self.max_length = max_length # max question word length
        self.c_num = 5 # max choice number

        # load and tokenize questions
        self.entries = _load_dataset(dataroot, name, task)
        if 'bert' in self.lang_model:
            self.tokenizer = AutoTokenizer.from_pretrained(bert_models[self.lang_model]) # For Bert
        self.tokenize()
        self.tensorize()

        # load image features
        h5_path = os.path.join(dataroot, 'patch_embeddings', feat_label, 'iconqa_%s_%s_%s.pth' % (name, task, feat_label))
        print('\nloading features from h5 file:', h5_path)
        self.features = torch.load(h5_path)
        self.num_patches = num_patches
        self.v_dim = list(self.features.values())[0].size()[1] # [num_patches,2048]
        print("visual feature dim:", self.v_dim)

        # load choice image features
        choice_h5_path = os.path.join(dataroot, 'img_choice_embeddings/',  choice_feat,
                               'iconqa_%s_%s_%s.pth' % (name, task, choice_feat))
        print('\nloading image choice features from h5 file:', choice_h5_path)
        self.choice_features = torch.load(choice_h5_path)

    def tokenize(self):
        """
        Tokenize the questions.
        This will add q_token in each entry of the dataset.
        """
        print('max token length is:', self.max_length)

        for entry in self.entries:
            if 'bert' in self.lang_model: # For Bert
                tokens = self.tokenizer(entry['question'])['input_ids']
                tokens = tokens[-self.max_length:]
                if len(tokens) < self.max_length:
                    tokens = tokens + [0] * (self.max_length - len(tokens))

            entry['q_token'] = tokens
            assert len(entry['q_token']) == self.max_length

    def tensorize(self):
        for entry in self.entries:
            question = torch.from_numpy(np.array(entry['q_token']))
            entry['q_token'] = question
            assert isinstance(entry['answer_label'], int) # 0-4

    def __getitem__(self, index):
        entry = self.entries[index]

        features = self.features[int(entry['image_id'])]

        question_id = entry['question_id']
        question = entry['q_token']
        answer = entry['answer']

        answer_label = entry['answer_label'] # 0-4
        assert answer == str(answer_label)

        target = torch.zeros(self.c_num)
        if answer_label in range(self.c_num):
            target[answer_label] = 1.0

        # for the choose_img sub-task
        choices = self.choice_features[int(entry['image_id'])]
        return features, question, choices, target, question_id

    def __len__(self):
        return len(self.entries)

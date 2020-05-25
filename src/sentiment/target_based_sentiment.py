# -*- coding: utf-8 -*-
# Adapted from:
# author: songyouwei <youwei0314@gmail.com>
# fixed: yangheng <yangheng@m.scnu.edu.cn>
# Copyright (C) 2018. All Rights Reserved.

import numpy as np
import torch
import torch.nn.functional as F


from .models.lcf_bert import LCF_BERT
from pytorch_transformers import BertModel
from .data_utils import Tokenizer4Bert
import argparse

def pad_and_truncate(sequence, maxlen, dtype='int64', padding='post', truncating='post', value=0):
    x = (np.ones(maxlen) * value).astype(dtype)
    if truncating == 'pre':
        trunc = sequence[-maxlen:]
    else:
        trunc = sequence[:maxlen]
    trunc = np.asarray(trunc, dtype=dtype)
    if padding == 'post':
        x[:len(trunc)] = trunc
    else:
        x[-len(trunc):] = trunc
    return x

def prepare_data(text_left, aspect, text_right, tokenizer):
    text_left = text_left.lower().strip()
    text_right = text_right.lower().strip()
    aspect = aspect.lower().strip()
    
    text_raw_indices = tokenizer.text_to_sequence(text_left + " " + aspect + " " + text_right)            
    aspect_indices = tokenizer.text_to_sequence(aspect)
    aspect_len = np.sum(aspect_indices != 0)
    text_bert_indices = tokenizer.text_to_sequence('[CLS] ' + text_left + " " + aspect + " " + text_right + ' [SEP] ' + aspect + " [SEP]")
    text_raw_bert_indices = tokenizer.text_to_sequence(
        "[CLS] " + text_left + " " + aspect + " " + text_right + " [SEP]")
    bert_segments_ids = np.asarray([0] * (np.sum(text_raw_indices != 0) + 2) + [1] * (aspect_len + 1))
    bert_segments_ids = pad_and_truncate(bert_segments_ids, tokenizer.max_seq_len)
    aspect_bert_indices = tokenizer.text_to_sequence("[CLS] " + aspect + " [SEP]")

    return text_bert_indices, bert_segments_ids, text_raw_bert_indices, aspect_bert_indices

def get_parameters():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='lcf_bert', type=str)
    parser.add_argument('--dataset', default='laptop', type=str, help='twitter, restaurant, laptop')
    parser.add_argument('--optimizer', default='adam', type=str)
    parser.add_argument('--initializer', default='xavier_uniform_', type=str)
    parser.add_argument('--learning_rate', default=2e-5, type=float, help='try 5e-5, 2e-5 for BERT, 1e-3 for others')
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--l2reg', default=0.01, type=float)
    parser.add_argument('--num_epoch', default=10, type=int, help='try larger number for non-BERT models')
    parser.add_argument('--batch_size', default=16, type=int, help='try 16, 32, 64 for BERT models')
    parser.add_argument('--log_step', default=5, type=int)
    parser.add_argument('--embed_dim', default=300, type=int)
    parser.add_argument('--hidden_dim', default=300, type=int)
    parser.add_argument('--bert_dim', default=768, type=int)
    parser.add_argument('--pretrained_bert_name', default='bert-base-uncased', type=str)
    parser.add_argument('--max_seq_len', default=250, type=int)
    parser.add_argument('--polarities_dim', default=3, type=int)
    parser.add_argument('--hops', default=3, type=int)
    parser.add_argument('--device', default=None, type=str, help='e.g. cuda:0')
    parser.add_argument('--seed', default=None, type=int, help='set seed for reproducibility')
    parser.add_argument('--valset_ratio', default=0, type=float,
                        help='set ratio between 0 and 1 for validation support')
    # The following parameters are only valid for the lcf-bert model
    parser.add_argument('--local_context_focus', default='cdm', type=str, help='local context focus mode, cdw or cdm')
    parser.add_argument('--SRD', default=3, type=int,
                        help='semantic-relative-distance, see the paper of LCF-BERT model')
    opt = parser.parse_args()
    return opt

class TargetSentimentAnalyzer:
    def __init__(self):
        model_classes = {
            # 'bert_spc': BERT_SPC,
            # 'aen_bert': AEN_BERT,
            'lcf_bert': LCF_BERT
        }

        # set your trained models here
        state_dict_paths = {
            'lcf_bert': 'models/lcf_bert_twitter_val_acc0.7283',
            # 'bert_spc': 'state_dict/bert_spc_laptop_val_acc0.268',
            # 'aen_bert': 'state_dict/aen_bert_laptop_val_acc0.2006'
        }

        self.opt = get_parameters()
        self.opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.opt.device = torch.device('cpu')

        print("TargetSentimentAnalyzer using device:",self.opt.device)

        self.tokenizer = Tokenizer4Bert(self.opt.max_seq_len, self.opt.pretrained_bert_name)
        self.bert = BertModel.from_pretrained(self.opt.pretrained_bert_name)
        self.model = model_classes[self.opt.model_name](self.bert, self.opt).to(self.opt.device)


        print('loading model {0} ...'.format(self.opt.model_name))
        self.model.load_state_dict(torch.load(state_dict_paths[self.opt.model_name]))
        self.model.eval()
        torch.autograd.set_grad_enabled(False)

    def get_sentiment(self, sentence, target):
        sentence_too_long = False

        location = sentence.find(target) # Find gets the first occurence of a substring
        if(location == -1):
            # No occurences, return None
            return (None, sentence_too_long)


        if(len(sentence) > self.opt.max_seq_len):
            sentence_too_long = True
            # print("To long sentence:" + str(len(sentence)) +", max sentence lenght: " + str(self.opt.max_seq_len) + " Trimming sentence. " +
            #  "Sencence: " + sentence)

            msl = self.opt.max_seq_len

            loc_tar = location + len(target)/2 # get the middle location of the target

            if(loc_tar - msl/2 >= 0):
                # Left side not hindered
                if(loc_tar + msl/2 <= len(sentence)):
                    # right side not hindered, trim evenly
                    sentence = sentence[max(int(loc_tar - msl/2),0):int(loc_tar + msl/2)]
                else:
                    # right side hindered
                    right = len(sentence) - loc_tar
                    left = msl - right
                    sentence = sentence[max(int(loc_tar - left),0):]
            else:
                # Left side hindered
                left = loc_tar
                right = msl - left

                if(location + msl/2 <= len(sentence)):
                    # right side not hindered
                    sentence = sentence[:int(loc_tar + right)]
                else:
                    # right side hindered
                    print("Error in sentence splitter, both sides are hindered, this cant be the case")
                    return (None, sentence_too_long)
            # print("Trimmed sentence:", sentence)
            # print("Sent lenght:",len(sentence))
            
            
            # return None

        # TODO: we work from first occurence if there is more than one target

        


        left = sentence[:location]
        right = sentence[location + len(target):]
        

        text_bert_indices, bert_segments_ids, text_raw_bert_indices, aspect_bert_indices = \
        prepare_data(left, target, right, self.tokenizer)
    
        text_bert_indices = torch.tensor([text_bert_indices], dtype=torch.int64).to(self.opt.device)
        bert_segments_ids = torch.tensor([bert_segments_ids], dtype=torch.int64).to(self.opt.device)
        text_raw_bert_indices = torch.tensor([text_raw_bert_indices], dtype=torch.int64).to(self.opt.device)
        aspect_bert_indices = torch.tensor([aspect_bert_indices], dtype=torch.int64).to(self.opt.device)
        if 'lcf' in self.opt.model_name:
            inputs = [text_bert_indices, bert_segments_ids, text_raw_bert_indices, aspect_bert_indices]
        # elif 'aen' in self.opt.model_name:
        #     inputs = [text_raw_bert_indices, aspect_bert_indices]
        # elif 'spc' in self.opt.model_name:
        #     inputs = [text_bert_indices, bert_segments_ids]
        outputs = self.model(inputs)
        t_probs = F.softmax(outputs, dim=-1).cpu().numpy()
        # print('t_probs = ', t_probs)
        sentiment = t_probs.argmax(axis=-1) - 1
        # print('aspect sentiment = ', sentiment)

        # print(sentiment[0])

        return (sentiment[0], sentence_too_long)
    
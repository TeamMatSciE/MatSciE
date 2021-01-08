import sys
sys.path.append('../')

import pickle
import re
import os
from tqdm import tqdm
import numpy as np
import json
import torch
import argparse
from typing import List, Union, Tuple
import tarfile
from sklearn import metrics
import spacy
from spacy.tokenizer import Tokenizer

from config import Span, Reader, Config, ContextEmb, lr_decay, simple_batching, evaluate_batch_insts, get_optimizer, write_results, batching_list_instances
from config import simple_batching, ContextEmb
from common import Instance, Sentence
from preprocess.get_elmo_vec import load_elmo, parse_sentence
from model import NNCRF
from create_dataset import CreateDataset
from allennlp.commands.elmo import ElmoEmbedder
from pytorch_pretrained_bert import BertTokenizer, BertModel

##Only annotated lines
# create_data = CreateDataset(False,True,False)
##All lines

def load_bert():
    tokenizer = BertTokenizer.from_pretrained('/home/souradip-pg/MtechProject/adaptabert/AdaptaBERT/NER/scibert/scibert_scivocab_cased', do_lower_case=False, do_basic_tokenize=False)
    bert_model = BertModel.from_pretrained('/home/souradip-pg/MtechProject/adaptabert/AdaptaBERT/NER/lm_output_annotated/').to(device)
    return tokenizer, bert_model

def get_bert_vec(words, tokenizer, bert_model):

    if len(words)> 300:
        words = words[:300]
    sentence=words[0]
    for word in words[1:]:
        sentence+=" " + word
    # print(len(words))

    sentence = "[CLS] " + sentence + " [SEP]"
    tokenized_text = tokenizer.tokenize(sentence)
    # print(len(tokenized_text))
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    segments_ids = [0] * len(tokenized_text)
    tokens_tensor = torch.tensor([indexed_tokens]).to(device)
    segments_tensors = torch.tensor([segments_ids]).to(device)
    model = bert_model
    model.eval()
    with torch.no_grad():
        # print(tokens_tensor.size(), segments_tensors.size())
        encoded_layers, _ = model(tokens_tensor, segments_tensors)
        summed_last_4_layers = torch.stack(encoded_layers[-4:]).sum(0)
        # print()
    vectors = summed_last_4_layers.cpu().detach().numpy()

    if mode == "average":
        result = np.average(vectors, 0)
        output = []
        for i,text in enumerate(tokenized_text):
            if text in ['[CLS]','[SEP]']:
                continue
            if not text.startswith('##'):
                output.append(result[i])
        assert(len(words) == len(output))
        return output
    elif mode == 'weighted_average':
        return np.swapaxes(vectors, 0, 1)
    elif mode == 'last':
        return vectors[-1, :, :]
    elif mode == 'all':
        return vectors
    else:
        return vectors

def str2bool(v):
    if isinstance(v, bool):
      return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        return False

create_data = CreateDataset(False,True,True)
paper_dir = '../../pdfs/data_18-01-2020/one/test'

def persection_conversion_bert(input_file,output_file, tokenizer, bert_model):
    file = open(input_file,"r+")
    new_file = open(output_file,"w+")
    lines = file.readlines()
    curr_section = ""
    count=0
    max=99999
    section_id = 0
    sentence_id = -1
    sentence_vec = []
    word_count = 0
    result = []
    prev_line = ""
    sentence = []

    for i in range(0,len(lines),1):
        line = lines[i]
        line = line.strip("\n")
        if line!="":
            words = [x for x in line.split()]
            if words[0]=="section" and prev_line == "":

                if sentence_id!=-1:

                    bert_vec = get_bert_vec(sentence, tokenizer, bert_model)

                    if len(bert_vec)!=word_count:
                        print(len(bert_vec), word_count)
                        temp = np.random.rand(word_count - len(bert_vec),len(bert_vec[0]))
                        bert_vec = np.concatenate((bert_vec,temp), axis=0)
                    sentence_vec.append(bert_vec)
                sentence_id += 1
                sentence = []
                word_count = 0
                
                if curr_section!="":
                    if curr_section!=words[1] or count>max:
                        #new section
                        new_file.write("\n")
                        new_file.write(line)
                        new_file.write("\n")
                        section_id += 1
                             
                        result.append(np.concatenate(sentence_vec, axis=0))
                        
                        sentence_vec = []
                        count=1
                        curr_section=words[1]
                    else:
                        #same section
                        count+=1
                else:
                    #first sentence
                    curr_section=words[1]
                    new_file.write(line)
                    new_file.write("\n")
                    count=1
                    
            else:
                if words[0]!="id":
                    sentence.append(words[0])
                    new_file.write(line)
                    new_file.write("\n")
                    word_count+=1
        prev_line = line

    # last sentence
    bert_vec = get_bert_vec(sentence, tokenizer, bert_model)

    if len(bert_vec)!=word_count:
        print(len(bert_vec), word_count)
        temp = np.random.rand(word_count - len(bert_vec),len(bert_vec[0]))
        bert_vec = np.concatenate((bert_vec,temp), axis=0)
    sentence_vec.append(bert_vec)

    #last section
    result.append(np.concatenate(sentence_vec, axis=0))
    new_file.close()

    parent = '/'.join(output_file.split('/')[:-1])+'/bert/'
    file_name = output_file.split('/')[-1]
    f = open(parent + file_name + '.bert.vec', 'wb')
    pickle.dump(result, f)
    f.close()


def persection_conversion(input_file,output_file):
    file = open(input_file,"r+")
    new_file = open(output_file,"w+")
    lines = file.readlines()
    curr_section = ""
    count=0
    max=99999
    for i in range(0,len(lines),1):
        line = lines[i]
        line = line.strip("\n")
        if line!="":
            words = [x for x in line.split()]
            if words[0]=="section" and words[1]!='O':
                if curr_section!="":
                    if curr_section!=words[1] or count>max:
                        new_file.write("\n")

                        new_file.write(line)
                        new_file.write("\n")
                        curr_section=words[1]
                        count=1
                    else:
                        #Section is continued
                        new_file.write('<UNK> O')
                        new_file.write('\n')
                        count+=1
                else:
                    curr_section=words[1]
                    new_file.write(line)
                    new_file.write("\n")
                    count=1
            else:
                # if words[0]!="id":
                new_file.write(line)
                new_file.write("\n")
    new_file.close()

def pdfannot_to_text(paper, fout, create_data):
    labels = []
    lines = []
    create_data.processFile(paper, labels, lines)
    # print(labels)
    create_data.tagLabels(labels, lines, fout)
    return labels


parser = argparse.ArgumentParser(description='Create test dataset.')
parser.add_argument("-paperDir",dest='paper_dir',action='store',help="directory that contains the parsed pdf files")
parser.add_argument("-persection",dest='persection',action='store', help="set true if persection model")
parser.add_argument("-context",dest='context',action='store', help="set elmo or bert")
args = parser.parse_args()

paper_dir = args.paper_dir
persection = str2bool(args.persection)
context = args.context

device = "cuda:0"
mode = "average"

if context=='bert':
    tokenizer, bert_model = load_bert()

for subdir, dirs, files in os.walk(paper_dir):
    for file in tqdm(files, desc="Files"):
        paper = os.path.join(subdir, file)
        # print(paper)
        if not paper.endswith('.json'):
            continue
        #print(paper)
        # if paper != '../../pdfs/test_fold1/04-09-2019-paper-19.pdf.json':
        #     continue

        if not persection:
            fout = open(paper[:-4] + 'txt','w')
            labels = pdfannot_to_text(paper, fout, create_data)
            fout.close()
        
        else:
            fout = open('test_pre.txt','w')
            labels = pdfannot_to_text(paper, fout, create_data)
            fout.close()
            if context == 'bert':
                persection_conversion_bert('test_pre.txt',paper[:-4] + 'txt', tokenizer, bert_model)
            else:
                persection_conversion('test_pre.txt',paper[:-4] + 'txt')

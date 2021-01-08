import sys
sys.path.append('../')

import pickle
import re
import os

from model import NNCRF
import torch
import argparse
import numpy as np

from config import simple_batching, ContextEmb
from typing import List, Union, Tuple
from common import Instance, Sentence
import tarfile
from sklearn.metrics import classification_report

from allennlp.commands.elmo import ElmoEmbedder
from get_elmo_vec import load_elmo, parse_sentence
from tqdm import tqdm


"""
Predictor usage example:

    sentence = "This is a sentence"
    # Or you can make a list of sentence:
    # sentence = ["This is a sentence", "This is the second sentence"]
    
    model_path = "english_model.tar.gz"
    model = NERPredictor(model_path)
    prediction = model.predict(sentence)
    print(prediction)

"""


class NERPredictor:
    """
    Usage:
    sentence = "This is a sentence"
    model_path = "model_files.tar.gz"
    model = Predictor(model_path)
    prediction = model.predict(sentence)
    """

    def __init__(self, model_archived_file:str, cuda_device: str = "cuda:0"):

        tar = tarfile.open(model_archived_file)
        tar.extractall()
        folder_name = tar.getnames()[0]
        tar.close()

        f = open(folder_name + "/config.conf", 'rb')
        self.conf = pickle.load(f)  # variables come out in the order you put them in
        # default batch size for conf is `10`
        f.close()
        device = torch.device(cuda_device)
        self.conf.device = device
        self.model = NNCRF(self.conf, print_info=False)
        self.model.load_state_dict(torch.load(folder_name + "/lstm_crf.m", map_location = device))
        self.model.eval()

        if self.conf.context_emb != ContextEmb.none:
            if cuda_device == "cpu":
                cuda_device = -1
            else:
                cuda_device = int(cuda_device.split(":")[1])
            self.elmo = load_elmo(cuda_device)

    def predict_insts(self, batch_insts_ids: Tuple) -> List[List[str]]:
        batch_max_scores, batch_max_ids = self.model.decode(batch_insts_ids)
        predictions = []
        for idx in range(len(batch_max_ids)):
            length = batch_insts_ids[1][idx]
            prediction = batch_max_ids[idx][:length].tolist()
            prediction = prediction[::-1]
            prediction = [self.conf.idx2labels[l] for l in prediction]
            predictions.append(prediction)
        return predictions

    def sent_to_insts(self, sentence: str) -> List[Instance]:
        words = sentence.split()
        return[Instance(Sentence(words))]

    def sents_to_insts(self, sentences: List[str]) -> List[Instance]:
        insts = []
        for sentence in sentences:
            words = sentence.split()
            insts.append(Instance(Sentence(words), None))
        return insts

    def create_batch_data(self, insts: List[Instance]):
        return simple_batching(self.conf, insts)

    def predict(self, sentences: Union[str, List[str]]):

        sents = [sentences] if isinstance(sentences, str) else sentences
        insts = self.sents_to_insts(sents)

        # for each in insts:
        #     print(each.input.words)

        # print(insts[0].input.words)
        self.conf.map_insts_ids(insts)
        if self.conf.context_emb != ContextEmb.none:
            read_parse_write(self.elmo, insts)
        test_batches = self.create_batch_data(insts)
        predictions = self.predict_insts(test_batches)

        # print(predictions)
        # if len(predictions) == 1:
        #     return predictions[0]
        # else:
        #     return predictions
        return predictions


def read_parse_write(elmo: ElmoEmbedder, insts: List[Instance], mode: str = "average") -> None:
    """
    Attach the instances into the sentence/
    :param elmo: ELMo embedder
    :param insts: List of instance
    :param mode: the mode of elmo vectors
    :return:
    """
    for inst in insts:
        vec = parse_sentence(elmo, inst.input.words, mode=mode)
        inst.elmo_vec = vec

    #sentences = [inst.input.words for inst in insts]
    #vectors = elmo.embed_sentences(sentences)

    #for i,vec in enumerate(vectors):
    #    vec = np.average(vec,0)
    #    insts[i].elmo_vec = vec


#with open('data/IKST_dataset_3line/test.txt') as f:

# paper_dir = '../../pdfs/data_18-01-2020/one/test'
# model_path = "../english_model/english_model.tar.gz"
# model = NERPredictor(model_path, cuda_device='cuda:1')


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Create test dataset.')
    parser.add_argument("-paperDir",dest='paper_dir',action='store',help="directory that contains the parsed pdf files")
    parser.add_argument("-modelPath",dest='model_path',action='store', help="path of the model")
    args = parser.parse_args()

    paper_dir = args.paper_dir
    model_path = args.model_path
    model = NERPredictor(model_path, cuda_device='cuda:0')


    cnt = 0
    for subdir, dirs, files in os.walk(paper_dir):
        files = [file for file in files if file.endswith('pdf.txt')]
        for file in tqdm(files, desc="Files"):
            paper = os.path.join(subdir, file)
            if not paper.endswith('pdf.txt'):
                continue

            cnt += 1
            #if cnt < 359:
            #    continue

            #if paper == 'data/arxiv_dataset_ultimatev2/1902.05128v1.A_nonlinear_hyperelasticity_model_for_single_layer_blue_phosphorus_based_on_ab_initio_calculations.pdf.txt':
            #     print(cnt)
            #     break
            predicted_paper = paper[:-3] + 'prediction'
#             if os.path.exists(predicted_paper):
#                 print(predicted_paper, "DONE")
#                 continue
            #continue
            print(paper)
            #continue

            try:

                with open(paper) as f:
                    sentences = []
                    labels = []
                    sections = []

                    sentence = ''
                    label = []
                    for line in f:
                        line = line.strip()
                        if line=='':
                            if sentence.strip()!='':
                                # sentences.append(sentence.strip())
                                # labels.append(label)
                                sentences.append(' '.join(sentence.strip().split()[1:]))
                                labels.append(label[1:])
                                sections.append(label[0])
                                
                            sentence = ''
                            label = []
                        else:
                            # sentence += re.sub('\d', '0', line.split()[0])+' '
                            sentence += line.split()[0]+' '
                            label.append(line.split()[1])
                            # sections.append(label[0])
                            
                    if sentence.strip()!='':
                        # sentences.append(sentence.strip())
                        # labels.append(label)
                        sentences.append(' '.join(sentence.strip().split()[1:]))
                        labels.append(label[1:])
                        sections.append(label[0])

                    # print(sentences[0])
                    if len(sentences) == 0:
                        continue
                    prediction = model.predict(sentences)
                    y_pred=[]
                    y_true=[]

                    fout = open(paper[:-3]+'prediction', 'w')
                    for i,words in enumerate(sentences):
                        words = words.split()
                        fout.write('section '+sections[i]+'\n')
                        # print(prediction[0])
                        for j,word in enumerate(words):
                            # print(i,j,word)
                            if prediction[i][j].startswith('S-'):
                                prediction[i][j] = 'B-'+prediction[i][j][2:]
                            if prediction[i][j].startswith('E-'):
                                prediction[i][j] = 'I-'+prediction[i][j][2:]
                            if "Material" in prediction[i][j]:
                                prediction[i][j] = prediction[i][j][:2] + "MATERIAL"
                            if "Material" in labels[i][j]:
                                labels[i][j] = labels[i][j][:2] + "MATERIAL"
                            
                            y_pred.append(prediction[i][j])
                            y_true.append(labels[i][j])
                            fout.write(word+' '+prediction[i][j]+' '+labels[i][j]+'\n')
                        fout.write('\n')

            except:
                print("ERROR", paper)

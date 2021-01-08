import argparse
import random
import numpy as np
from config import Reader, Config, ContextEmb, lr_decay, simple_batching, evaluate_batch_insts, get_optimizer, write_results, batching_list_instances
import time
from model.neuralcrf import NNCRF
import torch
from typing import List
from common import Instance
from termcolor import colored
import os
from config.utils import load_elmo_vec
import pickle
import tarfile
import shutil

def set_seed(opt, seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if opt.device.startswith("cuda"):
        print("using GPU...", torch.cuda.current_device())
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def parse_arguments(parser):
    ###Training Hyperparameters
    parser.add_argument('--device', type=str, default="cpu", choices=['cpu', 'cuda:0', 'cuda:1', 'cuda:2'],
                        help="GPU/CPU devices")
    parser.add_argument('--seed', type=int, default=42, help="random seed")
    parser.add_argument('--digit2zero', action="store_true", default=False,
                        help="convert the number to 0, make it true is better")
    parser.add_argument('--dataset', type=str, default="conll2003")
    parser.add_argument('--embedding_file', type=str, default="data/glove.6B.100d.txt",
                        help="we will be using random embeddings if file do not exist")
    parser.add_argument('--embedding_dim', type=int, default=100)
    parser.add_argument('--optimizer', type=str, default="sgd")
    parser.add_argument('--learning_rate', type=float, default=0.01)  ##only for sgd now
    parser.add_argument('--momentum', type=float, default=0.0)
    parser.add_argument('--l2', type=float, default=1e-8)
    parser.add_argument('--lr_decay', type=float, default=0)
    parser.add_argument('--batch_size', type=int, default=10, help="default batch size is 10 (works well)")
    parser.add_argument('--num_epochs', type=int, default=100, help="Usually we set to 10.")
    parser.add_argument('--train_num', type=int, default=-1, help="-1 means all the data")
    parser.add_argument('--dev_num', type=int, default=-1, help="-1 means all the data")
    parser.add_argument('--test_num', type=int, default=-1, help="-1 means all the data")
    

    ##model hyperparameter
    parser.add_argument('--num_layers',type=int,default=1)
    parser.add_argument('--model_folder', type=str, default="english_model", help="The name to save the model files")
    parser.add_argument('--hidden_dim', type=int, default=200, help="hidden size of the LSTM")
    parser.add_argument('--use_crf_layer', type=int, default=1, help="1 is for using crf layer, 0 for not using CRF layer", choices=[0,1])
    parser.add_argument('--dropout', type=float, default=0.5, help="dropout for embedding")
    parser.add_argument('--use_char_rnn', type=int, default=1, choices=[0, 1], help="use character-level lstm, 0 or 1")
    parser.add_argument('--context_emb', type=str, default="none", choices=["none", "elmo", "bert"],
                        help="contextual word embedding")

    args = parser.parse_args()
    for k in args.__dict__:
        print(k + ": " + str(args.__dict__[k]))
    return args


def train_model(config: Config, epoch: int, train_insts: List[Instance], dev_insts: List[Instance], test_insts: List[Instance]):
    model = NNCRF(config)
    optimizer = get_optimizer(config, model)
    train_num = len(train_insts)
    print("number of instances: %d" % (train_num))
    print(colored("[Shuffled] Shuffle the training instance ids", "red"))
    random.shuffle(train_insts)

    batched_data = batching_list_instances(config, train_insts)
    dev_batches = batching_list_instances(config, dev_insts)
    test_batches = batching_list_instances(config, test_insts)

    best_dev = [-1, 0]
    best_test = [-1, 0]

    model_folder = config.model_folder
    res_folder = "results"
    if os.path.exists(model_folder):
        raise FileExistsError(f"The folder {model_folder} exists. Please either delete it or create a new one "
                              f"to avoid override.")
    model_name = model_folder + "/lstm_crf.m".format()
    config_name = model_folder + "/config.conf"
    res_name = res_folder + "/lstm_crf.results".format()
    print("[Info] The model will be saved to: %s.tar.gz" % (model_folder))
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    if not os.path.exists(res_folder):
        os.makedirs(res_folder)

    for i in range(1, epoch + 1):
        epoch_loss = 0
        start_time = time.time()
        model.zero_grad()
        if config.optimizer.lower() == "sgd":
            optimizer = lr_decay(config, optimizer, i)
        for index in np.random.permutation(len(batched_data)):
            model.train()
            loss = model(*batched_data[index])
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
            model.zero_grad()
            loss.detach()

        end_time = time.time()
        print("Epoch %d: %.5f, Time is %.2fs" % (i, epoch_loss, end_time - start_time), flush=True)

        model.eval()
        dev_metrics = evaluate_model(config, model, dev_batches, "dev", dev_insts)
        test_metrics = evaluate_model(config, model, test_batches, "test", test_insts)
        if test_metrics[1][2] > best_test[0]:
            print("saving the best model...")
            best_dev[0] = dev_metrics[1][2]
            best_dev[1] = i
            best_test[0] = test_metrics[1][2]
            best_test[1] = i
            torch.save(model.state_dict(), model_name)
            # Save the corresponding config as well.
            f = open(config_name, 'wb')
            pickle.dump(config, f)
            f.close()
            print('Exact\n')
            print_report(test_metrics[-2])
            print('Overlap\n')
            print_report(test_metrics[-1])
            write_results(res_name, test_insts)
            print("Archiving the best Model...")
            with tarfile.open(model_folder + "/" + model_folder + ".tar.gz", "w:gz") as tar:
                tar.add(model_folder, arcname=os.path.basename(model_folder))
        model.zero_grad()
        
    print("Finished archiving the models")

    print("The best dev: %.2f" % (best_dev[0]))
    print("The corresponding test: %.2f" % (best_test[0]))
    print("Final testing.")
    model.load_state_dict(torch.load(model_name))
    model.eval()
    evaluate_model(config, model, test_batches, "test", test_insts)
    write_results(res_name, test_insts)

def print_report(dict_):
    print(" "*13+"  precision     recall       fscore    \n")
    for key in dict_:
        precision = dict_[key][0] * 1.0 / dict_[key][1] * 100 if dict_[key][1] != 0 else 0
        recall = dict_[key][0] * 1.0 / dict_[key][2] * 100 if dict_[key][2] != 0 else 0
        fscore = 2.0 * precision * recall / (precision + recall) if precision != 0 or recall != 0 else 0
        p = str(round(precision,2))
        r = str(round(recall,2))
        f = str(round(fscore,2))
        print(key+" "*(13-len(key))+" "*int((13-len(p))/2)+p + " "*int((13-len(p))/2)+" "*int((13-len(r))/2)+r+ " "*int((13-len(r))/2)+" "*int((13-len(f))/2)+f+ " "*int((13-len(f))/2))
    print("\n")


def evaluate_model(config: Config, model: NNCRF, batch_insts_ids, name: str, insts: List[Instance]):
    ## evaluation
    metrics_exact = np.asarray([0, 0, 0], dtype=int)
    metrics_overlap = np.asarray([0, 0, 0], dtype=int)

    dict_exact = {}
    dict_overlap = {}

    batch_id = 0
    batch_size = config.batch_size
    for batch in batch_insts_ids:
        one_batch_insts = insts[batch_id * batch_size:(batch_id + 1) * batch_size]
        batch_max_scores, batch_max_ids = model.decode(batch)
        results = evaluate_batch_insts(one_batch_insts, batch_max_ids, batch[-1], batch[1], config.idx2labels)

        metrics_exact += results[0]
        metrics_overlap += results[1]

        for key in results[2]:
            if key not in dict_exact:
                dict_exact[key] = [0,0,0]
            dict_exact[key][0] += results[2][key][0]
            dict_exact[key][1] += results[2][key][1]
            dict_exact[key][2] += results[2][key][2]

        for key in results[3]:
            if key not in dict_overlap:
                dict_overlap[key] = [0,0,0]
            dict_overlap[key][0] += results[3][key][0]
            dict_overlap[key][1] += results[3][key][1]
            dict_overlap[key][2] += results[3][key][2]

        batch_id += 1

    p_exact, total_predict, total_entity = metrics_exact[0], metrics_exact[1], metrics_exact[2]
    precision_exact = p_exact * 1.0 / total_predict * 100 if total_predict != 0 else 0
    recall_exact = p_exact * 1.0 / total_entity * 100 if total_entity != 0 else 0
    fscore_exact = 2.0 * precision_exact * recall_exact / (precision_exact + recall_exact) if precision_exact != 0 or recall_exact != 0 else 0
    print("[%s set - Exact] Precision: %.2f, Recall: %.2f, F1: %.2f" % (name, precision_exact, recall_exact, fscore_exact), flush=True)
    #print_report(dict_exact)

    p_overlap, total_predict, total_entity = metrics_overlap[0], metrics_overlap[1], metrics_overlap[2]
    precision_overlap = p_overlap * 1.0 / total_predict * 100 if total_predict != 0 else 0
    recall_overlap = p_overlap * 1.0 / total_entity * 100 if total_entity != 0 else 0
    fscore_overlap = 2.0 * precision_overlap * recall_overlap / (precision_overlap + recall_overlap) if precision_overlap != 0 or recall_overlap != 0 else 0
    print("[%s set - Overlap] Precision: %.2f, Recall: %.2f, F1: %.2f" % (name, precision_overlap, recall_overlap, fscore_overlap), flush=True)
    #print_report(dict_overlap)

    return [precision_exact, recall_exact, fscore_exact],[precision_overlap, recall_overlap, fscore_overlap], dict_exact, dict_overlap


def main():
    parser = argparse.ArgumentParser(description="LSTM CRF implementation")
    opt = parse_arguments(parser)
    conf = Config(opt)

    reader = Reader(conf.digit2zero)
    set_seed(opt, conf.seed)

    trains = reader.read_txt(conf.train_file, conf.train_num)
    devs = reader.read_txt(conf.dev_file, conf.dev_num)
    tests = reader.read_txt(conf.test_file, conf.test_num)

    if conf.context_emb != ContextEmb.none:
        print('Loading the ELMo vectors for all datasets.')
        conf.context_emb_size = load_elmo_vec(conf.train_file + "." + conf.context_emb.name + ".vec", trains)
        load_elmo_vec(conf.dev_file + "." + conf.context_emb.name + ".vec", devs)
        load_elmo_vec(conf.test_file + "." + conf.context_emb.name + ".vec", tests)

    conf.use_iobes(trains)
    conf.use_iobes(devs)
    conf.use_iobes(tests)
    conf.build_label_idx(trains+devs+tests)

    conf.build_word_idx(trains, devs, tests)
    conf.build_emb_table()

    conf.map_insts_ids(trains)
    conf.map_insts_ids(devs)
    conf.map_insts_ids(tests)

    print("num chars: " + str(conf.num_char))
    # print(str(config.char2idx))

    print("num words: " + str(len(conf.word2idx)))
    # print(config.word2idx)
    train_model(conf, conf.num_epochs, trains, devs, tests)


if __name__ == "__main__":
    main()

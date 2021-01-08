
import numpy as np
from overrides import overrides
from typing import List
from common import Instance
import torch


class Span:
    """
    A class of `Span` where we use it during evaluation.
    We construct spans for the convenience of evaluation.
    """
    def __init__(self, left: int, right: int, type: str,line:int=-1):
        """
        A span compose of left, right (inclusive) and its entity label.
        :param left:
        :param right: inclusive.
        :param type:
        """
        self.left = left
        self.right = right
        self.type = type
        self.line = line
    def __eq__(self, other):
        return self.left == other.left and self.right == other.right and self.type == other.type

    def __hash__(self):
        return hash((self.left, self.right, self.type))


def evaluate_batch_insts(batch_insts: List[Instance],
                         batch_pred_ids: torch.LongTensor,
                         batch_gold_ids: torch.LongTensor,
                         word_seq_lens: torch.LongTensor,
                         idx2label: List[str]):
    """
    Evaluate a batch of instances and handling the padding positions.
    :param batch_insts:  a batched of instances.
    :param batch_pred_ids: Shape: (batch_size, max_length) prediction ids from the viterbi algorithm.
    :param batch_gold_ids: Shape: (batch_size, max_length) gold ids.
    :param word_seq_lens: Shape: (batch_size) the length for each instance.
    :param idx2label: The idx to label mapping.
    :return: numpy array containing (number of true positive, number of all positive, number of true positive + number of false negative)
             You can also refer as (number of correctly predicted entities, number of entities predicted, number of entities in the dataset)
    """

    dict_exact = {}
    dict_overlap = {}

    p_exact = 0
    p_overlap = 0
    total_entity = 0
    total_predict = 0
    word_seq_lens = word_seq_lens.tolist()
    for idx in range(len(batch_pred_ids)):
        length = word_seq_lens[idx]
        output = batch_gold_ids[idx][:length].tolist()
        prediction = batch_pred_ids[idx][:length].tolist()
        prediction = prediction[::-1]
        output = [idx2label[l] for l in output]
        prediction =[idx2label[l] for l in prediction]
        batch_insts[idx].prediction = prediction
        #convert to span
        output_spans = set()
        start = -1
        for i in range(len(output)):
            if output[i].startswith("B-"):
                start = i
            if output[i].startswith("E-"):
                end = i
                output_spans.add(Span(start, end, output[i][2:]))
            if output[i].startswith("S-"):
                output_spans.add(Span(i, i, output[i][2:]))
        predict_spans = set()
        for i in range(len(prediction)):
            if prediction[i].startswith("B-"):
                start = i
            if prediction[i].startswith("E-"):
                end = i
                predict_spans.add(Span(start, end, prediction[i][2:]))
            if prediction[i].startswith("S-"):
                predict_spans.add(Span(i, i, prediction[i][2:]))

        total_entity += len(output_spans)
        total_predict += len(predict_spans)
        p_exact += len(predict_spans.intersection(output_spans))

        for each_output_span in output_spans:
            if each_output_span.type not in dict_overlap:
                dict_overlap[each_output_span.type] = [0,0,0]
                dict_exact[each_output_span.type] = [0,0,0]

            dict_exact[each_output_span.type][2] += 1
            dict_overlap[each_output_span.type][2] += 1

        for each_predicted_span in predict_spans:
            if each_predicted_span.type not in dict_overlap:
                dict_overlap[each_predicted_span.type] = [0,0,0]
                dict_exact[each_predicted_span.type] = [0,0,0]

            dict_exact[each_predicted_span.type][1] += 1
            dict_overlap[each_predicted_span.type][1] += 1

            if each_predicted_span in output_spans:
                dict_exact[each_predicted_span.type][0] += 1

            for each_output_span in output_spans:
                if each_predicted_span.type == each_output_span.type:
                    if each_output_span.left <= each_predicted_span.left <= each_output_span.right:
                        dict_overlap[each_predicted_span.type][0] += 1
                        p_overlap += 1
                        break;
                    if each_output_span.left <= each_predicted_span.right <= each_output_span.right:
                        dict_overlap[each_predicted_span.type][0] += 1
                        p_overlap += 1
                        break

                    if each_output_span.left>=each_predicted_span.left and each_output_span.right<=each_predicted_span.right:
                        dict_overlap[each_predicted_span.type][0] += 1
                        p_overlap += 1
                        break

    # In case you need the following code for calculating the p/r/f in a batch.
    # (When your batch is the complete dataset)
    # precision = p * 1.0 / total_predict * 100 if total_predict != 0 else 0
    # recall = p * 1.0 / total_entity * 100 if total_entity != 0 else 0
    # fscore = 2.0 * precision * recall / (precision + recall) if precision != 0 or recall != 0 else 0

    return np.asarray([p_exact, total_predict, total_entity], dtype=int), np.asarray([p_overlap, total_predict, total_entity], dtype=int), dict_exact, dict_overlap

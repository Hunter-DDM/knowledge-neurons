"""
BERT MLM runner
"""

import logging
import argparse
import math
import os
import torch
import random
import numpy as np
import json, jsonlines
import pickle
import time

import transformers
from transformers import BertTokenizer
from custom_bert import BertForMaskedLM
import torch.nn.functional as F

# set logger
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def example2feature(example, max_seq_length, tokenizer):
    """Convert an example into input features"""
    features = []
    tokenslist = []

    ori_tokens = tokenizer.tokenize(example[0])
    # All templates are simple, almost no one will exceed the length limit.
    if len(ori_tokens) > max_seq_length - 2:
        ori_tokens = ori_tokens[:max_seq_length - 2]

    # add special tokens
    tokens = ["[CLS]"] + ori_tokens + ["[SEP]"]
    base_tokens = ["[UNK]"] + ["[UNK]"] * len(ori_tokens) + ["[UNK]"]
    segment_ids = [0] * len(tokens)

    # Generate id and attention mask
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    baseline_ids = tokenizer.convert_tokens_to_ids(base_tokens)
    input_mask = [1] * len(input_ids)

    # Pad [PAD] tokens (id in BERT-base-cased: 0) up to the sequence length.
    padding = [0] * (max_seq_length - len(input_ids))
    input_ids += padding
    baseline_ids += padding
    segment_ids += padding
    input_mask += padding

    assert len(baseline_ids) == max_seq_length
    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    features = {
        'input_ids': input_ids,
        'input_mask': input_mask,
        'segment_ids': segment_ids,
        'baseline_ids': baseline_ids,
    }
    tokens_info = {
        "tokens":tokens,
        "relation":example[2],
        "gold_obj":example[1],
        "pred_obj": None
    }
    return features, tokens_info


def scaled_input(emb, batch_size, num_batch):
    # emb: (1, ffn_size)
    baseline = torch.zeros_like(emb)  # (1, ffn_size)

    num_points = batch_size * num_batch
    step = (emb - baseline) / num_points  # (1, ffn_size)

    res = torch.cat([torch.add(baseline, step * i) for i in range(num_points)], dim=0)  # (num_points, ffn_size)
    return res, step[0]


def convert_to_triplet_ig(ig_list):
    ig_triplet = []
    ig = np.array(ig_list)  # 12, 3072
    max_ig = ig.max()
    for i in range(ig.shape[0]):
        for j in range(ig.shape[1]):
            if ig[i][j] >= max_ig * 0.1:
                ig_triplet.append([i, j, ig[i][j]])
    return ig_triplet


def main():
    parser = argparse.ArgumentParser()

    # Basic parameters
    parser.add_argument("--data_path",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data path. Should be .json file for the MLM task. ")
    parser.add_argument("--tmp_data_path",
                        default=None,
                        type=str,
                        help="Temporary input data path. Should be .json file for the MLM task. ")
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                            "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--output_prefix",
                        default=None,
                        type=str,
                        required=True,
                        help="The output prefix to indentify each running of experiment. ")

    # Other parameters
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                            "Sequences longer than this will be truncated, and sequences shorter \n"
                            "than this will be padded.")
    parser.add_argument("--do_lower_case",
                        default=False,
                        action='store_true',
                        help="Set this flag if you are using an uncased model")
    parser.add_argument("--no_cuda",
                        default=False,
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--gpus",
                        type=str,
                        default='0',
                        help="available gpus id")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument("--debug",
                        type=int,
                        default=-1,
                        help="How many examples to debug. -1 denotes no debugging")
    parser.add_argument("--pt_relation",
                        type=str,
                        default=None,
                        help="Relation to calculate on clusters")

    # parameters about integrated grad
    parser.add_argument("--get_pred",
                        action='store_true',
                        help="Whether to get prediction results.")
    parser.add_argument("--get_ig_pred",
                        action='store_true',
                        help="Whether to get integrated gradient at the predicted label.")
    parser.add_argument("--get_ig_gold",
                        action='store_true',
                        help="Whether to get integrated gradient at the gold label.")
    parser.add_argument("--get_base",
                        action='store_true',
                        help="Whether to get base values. ")
    parser.add_argument("--batch_size",
                        default=16,
                        type=int,
                        help="Total batch size for cut.")
    parser.add_argument("--num_batch",
                        default=10,
                        type=int,
                        help="Num batch of an example.")

    # parse arguments
    args = parser.parse_args()

    # set device
    if args.no_cuda or not torch.cuda.is_available():
        device = torch.device("cpu")
        n_gpu = 0
    elif len(args.gpus) == 1:
        device = torch.device("cuda:%s" % args.gpus)
        n_gpu = 1
    else:
        # !!! to implement multi-gpus
        pass
    print("device: {} n_gpu: {}, distributed training: {}".format(device, n_gpu, bool(n_gpu > 1)))

    # set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    # save args
    os.makedirs(args.output_dir, exist_ok=True)
    json.dump(args.__dict__, open(os.path.join(args.output_dir, args.output_prefix + '.args.json'), 'w'), sort_keys=True, indent=2)

    # init tokenizer
    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    # Load pre-trained BERT
    print("***** CUDA.empty_cache() *****")
    torch.cuda.empty_cache()
    model = BertForMaskedLM.from_pretrained(args.bert_model)
    model.to(device)

    # data parallel
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)
    model.eval()

    # prepare eval set
    if os.path.exists(args.tmp_data_path):
        with open(args.tmp_data_path, 'r') as f:
            eval_bag_list_perrel = json.load(f)
    else:
        with open(args.data_path, 'r') as f:
            eval_bag_list_all = json.load(f)
        # split bag list into relations
        eval_bag_list_perrel = {}
        for bag_idx, eval_bag in enumerate(eval_bag_list_all):
            bag_rel = eval_bag[0][2].split('(')[0]
            if bag_rel not in eval_bag_list_perrel:
                eval_bag_list_perrel[bag_rel] = []
            if len(eval_bag_list_perrel[bag_rel]) >= args.debug:
                continue
            eval_bag_list_perrel[bag_rel].append(eval_bag)
        with open(args.tmp_data_path, 'w') as fw:
            json.dump(eval_bag_list_perrel, fw, indent=2)

    # evaluate args.debug bags for each relation
    for relation, eval_bag_list in eval_bag_list_perrel.items():
        if args.pt_relation is not None and relation != args.pt_relation:
            continue
        # record running time
        tic = time.perf_counter()
        with jsonlines.open(os.path.join(args.output_dir, args.output_prefix + '-' + relation + '.rlt' + '.jsonl'), 'w') as fw:
            for bag_idx, eval_bag in enumerate(eval_bag_list):
                res_dict_bag = []
                for eval_example in eval_bag:
                    eval_features, tokens_info = example2feature(eval_example, args.max_seq_length, tokenizer)
                    # convert features to long type tensors
                    baseline_ids, input_ids, input_mask, segment_ids = eval_features['baseline_ids'], eval_features['input_ids'], eval_features['input_mask'], eval_features['segment_ids']
                    baseline_ids = torch.tensor(baseline_ids, dtype=torch.long).unsqueeze(0)
                    input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)
                    input_mask = torch.tensor(input_mask, dtype=torch.long).unsqueeze(0)
                    segment_ids = torch.tensor(segment_ids, dtype=torch.long).unsqueeze(0)
                    baseline_ids = baseline_ids.to(device)
                    input_ids = input_ids.to(device)
                    input_mask = input_mask.to(device)
                    segment_ids = segment_ids.to(device)

                    # record real input length
                    input_len = int(input_mask[0].sum())

                    # record [MASK]'s position
                    tgt_pos = tokens_info['tokens'].index('[MASK]')

                    # record various results
                    res_dict = {
                        'pred': [],
                        'ig_pred': [],
                        'ig_gold': [],
                        'base': []
                    }

                    # original pred prob
                    if args.get_pred:
                        _, logits = model(input_ids=input_ids, attention_mask=input_mask, token_type_ids=segment_ids, tgt_pos=tgt_pos, tgt_layer=0)  # (1, n_vocab)
                        base_pred_prob = F.softmax(logits, dim=1)  # (1, n_vocab)
                        res_dict['pred'].append(base_pred_prob.tolist())

                    for tgt_layer in range(model.bert.config.num_hidden_layers):
                        ffn_weights, logits = model(input_ids=input_ids, attention_mask=input_mask, token_type_ids=segment_ids, tgt_pos=tgt_pos, tgt_layer=tgt_layer)  # (1, ffn_size), (1, n_vocab)
                        pred_label = int(torch.argmax(logits[0, :]))  # scalar
                        gold_label = tokenizer.convert_tokens_to_ids(tokens_info['gold_obj'])
                        tokens_info['pred_obj'] = tokenizer.convert_ids_to_tokens(pred_label)
                        scaled_weights, weights_step = scaled_input(ffn_weights, args.batch_size, args.num_batch)  # (num_points, ffn_size), (ffn_size)
                        scaled_weights.requires_grad_(True)

                        # integrated grad at the pred label for each layer
                        if args.get_ig_pred:
                            ig_pred = None
                            for batch_idx in range(args.num_batch):
                                batch_weights = scaled_weights[batch_idx * args.batch_size:(batch_idx + 1) * args.batch_size]
                                _, grad = model(input_ids=input_ids, attention_mask=input_mask, token_type_ids=segment_ids, tgt_pos=tgt_pos, tgt_layer=tgt_layer, tmp_score=batch_weights, tgt_label=pred_label)  # (batch, n_vocab), (batch, ffn_size)
                                grad = grad.sum(dim=0)  # (ffn_size)
                                ig_pred = grad if ig_pred is None else torch.add(ig_pred, grad)  # (ffn_size)
                            ig_pred = ig_pred * weights_step  # (ffn_size)
                            res_dict['ig_pred'].append(ig_pred.tolist())

                        # integrated grad at the gold label for each layer
                        if args.get_ig_gold:
                            ig_gold = None
                            for batch_idx in range(args.num_batch):
                                batch_weights = scaled_weights[batch_idx * args.batch_size:(batch_idx + 1) * args.batch_size]
                                _, grad = model(input_ids=input_ids, attention_mask=input_mask, token_type_ids=segment_ids, tgt_pos=tgt_pos, tgt_layer=tgt_layer, tmp_score=batch_weights, tgt_label=gold_label)  # (batch, n_vocab), (batch, ffn_size)
                                grad = grad.sum(dim=0)  # (ffn_size)
                                ig_gold = grad if ig_gold is None else torch.add(ig_gold, grad)  # (ffn_size)
                            ig_gold = ig_gold * weights_step  # (ffn_size)
                            res_dict['ig_gold'].append(ig_gold.tolist())

                        # base ffn_weights for each layer
                        if args.get_base:
                            res_dict['base'].append(ffn_weights.squeeze().tolist())

                    if args.get_ig_gold:
                        res_dict['ig_gold'] = convert_to_triplet_ig(res_dict['ig_gold'])
                    if args.get_base:
                        res_dict['base'] = convert_to_triplet_ig(res_dict['base'])

                    res_dict_bag.append([tokens_info, res_dict])

                fw.write(res_dict_bag)

        # record running time
        toc = time.perf_counter()
        print(f"***** Relation: {relation} evaluated. Costing time: {toc - tic:0.4f} seconds *****")


if __name__ == "__main__":
    main()
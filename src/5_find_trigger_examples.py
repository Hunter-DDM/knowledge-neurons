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
import re
from nltk.tokenize import word_tokenize

import transformers
from transformers import BertTokenizer
from custom_bert import BertForMaskedLM
import torch.nn.functional as F

# set logger
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def example2feature_distant(example, max_seq_length, tokenizer):
    """Convert an example into input features"""
    features = []
    tokenslist = []

    ori_tokens = tokenizer.tokenize(example)
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


def check_h_t(sent, h, t):
    sent = re.sub(h, '', sent)
    if re.search(f'(?<![a-zA-Z]){t}(?![a-zA-Z])', sent):
        return True
    else:
        return False


def check_t(sent, t):
    if re.search(f'(?<![a-zA-Z]){t}(?![a-zA-Z])', sent):
        return True
    else:
        return False


def mask_process_th(bag_th, t, h):
    new_bag_th = []
    pattern_h = re.sub(r'\)', r'\)', h)
    pattern_h = re.sub(r'\(', r'\(', pattern_h)
    pattern_h = re.sub(r'\$', r'\$', pattern_h)
    pattern_h = re.sub(r'\.', r'\.', pattern_h)
    pattern_t = re.sub(r'\)', r'\)', t)
    pattern_t = re.sub(r'\(', r'\(', pattern_t)
    pattern_t = re.sub(r'\$', r'\$', pattern_t)
    pattern_t = re.sub(r'\.', r'\.', pattern_t)
    for i in range(len(bag_th)):
        sent = bag_th[i]
        pad_sent = re.sub(pattern_h, '@' * len(h), sent)
        mask_sent = re.sub(f'(?<![a-zA-Z]){pattern_t}(?![a-zA-Z])', '[MASK]', pad_sent, count=1)
        new_sent = re.sub('@' * len(h), h, mask_sent)
        new_bag_th.append(new_sent)
    return new_bag_th


def mask_process_h(bag_h, h):
    new_bag_h = []
    pattern_h = re.sub(r'\)', r'\)', h)
    pattern_h = re.sub(r'\(', r'\(', pattern_h)
    pattern_h = re.sub(r'\$', r'\$', pattern_h)
    pattern_h = re.sub(r'\.', r'\.', pattern_h)
    for i in range(len(bag_h)):
        sent = bag_h[i]
        sent = re.sub(pattern_h, '@' * len(h), sent)
        words = sent.split(' ')
        tgt_pos = random.randint(1, min(len(words), 50)) - 1
        # do not mask h, and prevent it from infinite loop
        repeat_num = 0
        while words[tgt_pos] == '@' * len(h):
            tgt_pos = random.randint(1, min(len(words), 50)) - 1
            repeat_num += 1
            if repeat_num > 3:
                break
        words[tgt_pos] = '[MASK]'
        new_sent = ' '.join(words)
        new_sent = re.sub('@' * len(h), h, new_sent)
        new_bag_h.append(new_sent)
    return new_bag_h


def mask_process_rd(bag_rd):
    new_bag_rd = []
    for i in range(len(bag_rd)):
        sent = bag_rd[i]
        words = sent.split(' ')
        tgt_pos = random.randint(1, min(len(words), 50)) - 1
        words[tgt_pos] = '[MASK]'
        new_sent = ' '.join(words)
        new_bag_rd.append(new_sent)
    return new_bag_rd


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
    parser.add_argument("--distant_data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="Distant supervision input data dir. ")
    parser.add_argument("--query_pairs_path",
                        default=None,
                        type=str,
                        required=True,
                        help="Path of query pairs to produce distant data. Should be .json file. ")
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                        "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--kn_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The directory where important positions are stored.")

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
    logger.info("device: {} n_gpu: {}, distributed training: {}".format(device, n_gpu, bool(n_gpu > 1)))

    # set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    # save args
    os.makedirs(args.output_dir, exist_ok=True)

    # init tokenizer
    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    # Load pre-trained BERT
    logger.info("***** CUDA.empty_cache() *****")
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

    d_eval_bag_list_perrel = {}
    for filename in os.listdir(args.distant_data_dir):
        if not filename.startswith('final_distant_allbags_'):
            continue
        if not filename.endswith('_new.json'):
            continue
        rel = filename.split('.')[0].split('_')[-2]
        with open(os.path.join(args.distant_data_dir, filename), 'r') as f:
            d_eval_bag_list = json.load(f)
            d_eval_bag_list_perrel[rel] = d_eval_bag_list

    with open(args.query_pairs_path, 'r', encoding='utf-8') as f:
        query_pairs = json.load(f)

    all_sentence = []
    for rel, d_eval_bag_list in d_eval_bag_list_perrel.items():
        for bag_idx, d_eval_bag in enumerate(d_eval_bag_list):
            all_sentence.extend(d_eval_bag['th_bags'] + d_eval_bag['h_bags'])


    for filename in os.listdir(args.kn_dir):
        if not filename.startswith('kn_bag-'):
            continue
        relation = filename.split('.')[0].split('-')[-1]
        if relation != 'P740' and relation != 'P937':
            continue
        with open(os.path.join(args.kn_dir, filename), 'r') as fr:
            kn_bag_list = json.load(fr)
        # record running time
        tic = time.perf_counter()
        for bag_idx, kn_bag in enumerate(kn_bag_list):
            eval_bag_th = d_eval_bag_list_perrel[relation][bag_idx]['th_bags']
            h, t = query_pairs[relation][bag_idx][0], query_pairs[relation][bag_idx][1]
            if len(eval_bag_th) == 0:
                continue
            if len(kn_bag) == 0:
                continue
            eval_bag_th = mask_process_th(eval_bag_th, t, h)
            # eval_bag_rd = random.sample(all_sentence, len(eval_bag_th) * 10)
            # eval_bag_rd = mask_process_rd(eval_bag_rd)
            score_and_sentence = []
            # # ==================== eval th+rd examples ===================
            # for eval_example in eval_bag_th + eval_bag_rd:
            # ==================== eval th examples ===================
            for eval_example in eval_bag_th:
                eval_features, tokens_info = example2feature_distant(eval_example, args.max_seq_length, tokenizer)
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

                # record [MASK]'s position
                tgt_pos = tokens_info['tokens'].index('[MASK]')

                imp_weights = model(input_ids=input_ids, attention_mask=input_mask, token_type_ids=segment_ids, tgt_pos=tgt_pos, tgt_layer=0, imp_pos=kn_bag, imp_op='return')  # list, (n_kn)
                score_and_sentence.append([sum(imp_weights), ' '.join(tokens_info['tokens'])])
            score_and_sentence = sorted(score_and_sentence, key=lambda x:-x[0])
            with open(os.path.join(args.output_dir, f'{relation}-{bag_idx}.txt'), 'w') as f:
                f.write('Triplet:\n')
                f.write(f'({h}, {relation}, {t})\n')
                f.write('Kneurons:\n')
                for kn in kn_bag:
                    f.write(f'{kn[0]} {kn[1]}\n')
                f.write('Top-10 Trigger examples:\n')
                for s_s in score_and_sentence[:20]:
                    f.write(f'{s_s[0]}\t{s_s[1]}\n')
        # record running time
        toc = time.perf_counter()
        logger.info(f"***** Relation: {relation} evaluated. Costing time: {toc - tic:0.4f} seconds *****")


if __name__ == "__main__":
    main()
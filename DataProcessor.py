# -*- coding: utf-8 -*-
# @Time    : 11/29/2022
# @Author  : Jing Zhang


import json
import os
import re
from typing import List
import copy
import argparse


def load_json_data(data_file):
    if not os.path.exists(data_file):
        raise Exception(f'{data_file} does not exist')
    with open(data_file, 'r', encoding='utf-8') as fr:
        data = json.load(fr)
    return data


def save_json_data(data, filepath):
    folder = os.path.dirname(filepath)
    if not os.path.exists(folder):
        os.makedirs(folder)

    with open(filepath, 'w', encoding='utf-8') as fw:
        fw.write(json.dumps(data, indent=2, ensure_ascii=False))


def extract_html_web_marks(text: str):
    """
    extract spans of html websites, web marks, ...
    :param text: input sentence
    :return: a list of spans (start, end) indicating the positions for html websites, web marks, ...
    """
    html_pattern = re.compile(r'https?://.*?(?=[\u4e00-\u9fff])')
    web_pattern = re.compile(r'(<sub>)|(</sub>)')
    fig_pattern = re.compile(r'（图.*?）')
    matches = []
    patterns = [html_pattern, web_pattern, fig_pattern]
    for pattern in patterns:
        for m in pattern.finditer(text):
            matches.append((m.start(), m.end()))
    # sort and remove special cases
    matches.sort(key=lambda x: (x[0], x[1]))
    new_matches = []
    for i, (start, end) in enumerate(matches):
        if len(new_matches) == 0:
            new_matches.append((start, end))
        elif end > new_matches[-1][1]:
            new_matches.append((start, end))

    return new_matches


def span_2_index(spans, text_len):
    """
    convert spans into position mappings, for later usage,
    words within spans are to be removed
    :param spans: a list of tuples (start, end)
    :param text_len: initial text length
    :return: position mappings indicating the position changes.
    """
    pre2cur_index = [-1] * text_len
    pre = 0
    cur = 0
    for start, end in spans:
        if pre < start:
            pre2cur_index[pre:start] = range(cur, (cur + start - pre))
            cur = cur + start - pre
        pre = end
    pre2cur_index[pre:] = range(cur, (cur + text_len - pre))

    new_len = cur + text_len - pre
    cur2pre_index = [-1] * new_len
    for i, num in enumerate(pre2cur_index):
        if num >= 0:
            cur2pre_index[num] = i

    return pre2cur_index, cur2pre_index


def extract_text_on_spans(text, spans):
    """
    update text according to provided spans,
    only retain words not in the given spans,
    :param text: initial text
    :param spans: a list of tuples (start, end)
    :return: new text
    """
    pre = 0
    res = ""
    for start, end in spans:
        if pre < start:
            res += text[pre:start]
        pre = end
    res += text[pre:]
    return res


def split_text(text: str, max_len: int, split_pat=r'([，,。？?！!；;][”"]?)'):
    """
    split text into sub-sentences based on punctuations.
    :param text: input sentence
    :param max_len: maximum sentence length; input sentence is split if longer than max_len
    :param split_pat: regulation pattern
    :return: all sub-sentences, and corresponding start positions
    """

    if len(text) <= max_len:
        return [text], [0]

    segments = re.split(split_pat, text)
    sentences = []
    for i in range(0, len(segments) - 1, 2):
        sentences.append(segments[i] + segments[i + 1])
    if segments[-1]:
        sentences.append(segments[-1])
    n_sen = len(sentences)
    sent_lens = [len(s) for s in sentences]

    all_sens = []
    for i in range(n_sen):
        length = 0
        sub = []
        for j in range(i, n_sen):
            if length + sent_lens[j] <= max_len or not sub:
                sub.append(j)
                length += sent_lens[j]
            else:
                break
        all_sens.append(sub)

        if j == n_sen - 1:
            if sub[-1] != j:
                all_sens.append(sub[1:] + [j])
            break

    if len(all_sens) == 1:
        return [text], [0]

    DG = {}
    N = len(all_sens)
    for k in range(N):
        tmp = list(range(k + 1, min(all_sens[k][-1] + 1, N)))
        if not tmp:
            tmp.append(k + 1)
        DG[k] = tmp

    routes = {N: (0, -1)}
    for i in range(N - 1, -1, -1):
        tmp = []
        for j in DG[i]:
            cross = set(all_sens[i]) & (set(all_sens[j]) if j < len(all_sens) else set())
            w_ij = sum([sent_lens[k] for k in cross]) ** 2
            w_j = routes[j][0]
            w_i_ = w_ij + w_j
            tmp.append((w_i_, j))
        routes[i] = min(tmp)

    sub_texts, sub_starts = [''.join([sentences[i] for i in all_sens[0]])], [0]
    k = 0
    while True:
        k = routes[k][1]
        sub_texts.append(''.join([sentences[i] for i in all_sens[k]]))
        sub_starts.append(sum(sent_lens[: all_sens[k][0]]))
        if k == N - 1:
            break

    return sub_texts, sub_starts


class DataProcessor:
    def __init__(self, is_test=False):
        self.is_test = is_test

        # revised according to the format of data files
        self.entities = 'entities'
        self.text = 'text'
        self.start = 'start_idx'
        self.end = 'end_idx'
        self.type = 'type'
        self.entity = 'entity'

        # variables
        self.cur2pre_index = []
        self.pre2cur_index = []
        self.ori_texts = []
        self.sen_labels = []
        self.starts = []

    def update_entities(self, entities, pre2cur_index):
        """
        Update start, end, text information of each entity according to change of the positions
        :param entities: a list of entities
        :param pre2cur_index: a list, pre2cur_index[i] indicates the current position for initial position at index i.
        :return: updated entities
        """
        new_entities = []
        for entity in entities:
            start = entity[self.start]
            end = entity[self.end]
            type_ = entity[self.type]
            text = entity[self.entity]

            start_ = -1
            end_ = -1
            text_ = ""
            for pos in range(start, end + 1):
                idx = pos - start
                if pre2cur_index[pos] >= 0:
                    if start_ == -1:
                        start_ = pre2cur_index[pos]
                    end_ = pre2cur_index[pos]
                    text_ += text[idx]
            if len(text_) > 0:
                new_entities.append({
                    self.start: start_,
                    self.end: end_,
                    self.type: type_,
                    self.entity: text_
                })

        return new_entities

    def split_entity_on_marks(self, entities: List[dict], kinds=['sym'], len_trd=20):
        """
        split long entity based on punctuations;
        Only entities of the specified types and longer than the threshold are split.
        :param entities: a list of entities
        :param kinds: the entity types to be implemented
        :param len_trd: entity length threshold
        :return: new entities with split entities
        """
        marks = ['。', '？', '！', '?', '!', ';', '；', ',', '，']
        pattern = re.compile(r'[，。？！,?!;；]')
        new_entities = []
        for entity in entities:
            text = entity[self.entity]
            type_ = entity[self.type]
            start = entity[self.start]
            if not (len(text) >= len_trd and type_ in kinds and any([mark in text for mark in marks])):
                new_entities.append(entity)
            else:
                pieces = re.split(pattern, text)
                pre = start
                for piece in pieces:
                    if len(piece) > 0:
                        start_ = pre
                        end_ = start_ + len(piece) - 1
                        new_entities.append({
                            self.start: start_,
                            self.end: end_,
                            self.type: type_,
                            self.entity: piece
                        })
                        pre = end_ + 2
                    else:
                        pre += 1

        return new_entities

    def split_sentence_update_entities(self, samples: List[dict], max_len=128):
        """
        split long text input and deal with the corresponding entities
        :param samples: a list of entity dictionaries
        :param max_len: maximum sentence length
        :return:
        """
        sen_labels = []
        starts = []
        new_samples = []
        for sen_id, sample in enumerate(samples):
            text = sample[self.text][:]
            if len(text) <= max_len - 2:
                sen_labels.append(sen_id)
                starts.append(0)
                new_samples.append(sample)
                continue
            else:
                sub_texts, sub_starts = split_text(text, max_len=max_len - 2)
                sen_labels += [sen_id] * len(sub_starts)
                starts += sub_starts
                if not self.is_test:
                    entities = sample[self.entities]
                    entities.sort(key=lambda x: x[self.start])
                    idx = 0
                    for sub_text, sub_start in zip(sub_texts, sub_starts):
                        tmp = []
                        while idx < len(entities):
                            entity = entities[idx]
                            if entity[self.end] <= sub_start+len(sub_text)-1:
                                start_new = 0 if entity[self.start] < sub_start else (entity[self.start] - sub_start)
                                tmp.append({
                                    self.start: start_new,
                                    self.end: entity[self.end] - sub_start,
                                    self.type: entity[self.type],
                                    self.entity: sub_text[start_new: (entity[self.end] - sub_start + 1)]
                                })
                                idx += 1
                            else:
                                break

                        new_samples.append({
                            self.text: sub_text,
                            self.entities: tmp
                        })
                else:
                    for sub_text in sub_texts:
                        new_samples.append({
                            self.text: sub_text,
                            self.entities: []
                        })

        self.sen_labels = sen_labels
        self.starts = starts
        return new_samples

    def recover_test_entities(self, samples: List[dict]):
        """
        for test file, if the initial text is processed,
        we need to recover the test results to match the initial positions
        :param samples:  a list of test results
        :return: test results matched to initial text input.
        """
        test_results = []
        sen_labels, starts = self.sen_labels, self.starts
        cur2pre_index = self.cur2pre_index
        if self.is_test:
            assert len(self.sen_labels) == len(self.starts)
            assert len(self.starts) == len(samples)
            # concatenate short sub-sentences
            cur = 0
            for idx, ori_text in enumerate(self.ori_texts):
                entities = []
                while cur < len(sen_labels) and sen_labels[cur] == idx:
                    start = starts[cur]
                    ents = samples[cur][self.entities]
                    for ent in ents:
                        ent[self.start] += start
                        ent[self.end] += start
                        entities.append(ent)
                    cur += 1
                test_results.append({
                    self.text: ori_text,
                    self.entities: entities
                })

            # convert to initial sentences
            for sample, cur2pre in zip(test_results, cur2pre_index):
                text = sample[self.text]
                ents = sample[self.entities]
                for ent in ents:
                    start_ = cur2pre[ent[self.start]]
                    end_ = cur2pre[ent[self.end]]
                    assert start_ <= end_, 'original start should be lower than end.'
                    ent[self.entity] = text[start_:(end_+1)]
                    ent[self.start] = start_
                    ent[self.end] = end_

            # remove repeated entities
            for sample in test_results:
                ents = []
                sample[self.entities].sort(key=lambda x: (x[self.start], x[self.end]))
                for idx, ent in enumerate(sample[self.entities]):
                    if len(ents) == 0 or ent != ents[-1]:
                        ents.append(ent)
                sample[self.entities] = ents

            return test_results

    def check_generated_samples(self, samples: List):
        """
        To check the validity of generated entities
        :param samples: a list of {"text": ..; "entities": ...}
        :return: True/False, False indicates some entities are problematic.
        """
        for sample in samples:
            text = sample[self.text]
            entities = sample[self.entities]
            for ent in entities:
                if text[ent[self.start]:(ent[self.end]+1)] != ent[self.entity]:
                    print(f'Generated case with text: {text} is problematic.')
                    return False
        return True

    def preprocess(self, data_path, save_path, max_len, operators=['clean_web', 'split_entity', 'split_text']):
        """
        this function is to load corpus file and preprocess it and then save
        :param data_path: file path of data
        :param save_path: file path to save the processed data
        :param max_len: maximum sentence length
        :param operators: a list of clean operators to be implemented
        :return:
        """
        samples = load_json_data(data_path)
        self.ori_texts = [sample[self.text] for sample in samples]
        self.pre2cur_index = []
        self.cur2pre_index = []
        new_samples = []

        for sample in samples:
            ini_text = sample[self.text]
            ini_entities = sample[self.entities]
            new_text = ini_text[:]
            new_entities = copy.deepcopy(ini_entities)

            # process html & web marks
            if 'clean_web' in operators:
                spans = extract_html_web_marks(ini_text)
                new_text = extract_text_on_spans(ini_text, spans)
                pre2cur_index, cur2pre_index = span_2_index(spans, len(ini_text))
                self.pre2cur_index.append(pre2cur_index)
                self.cur2pre_index.append(cur2pre_index)
                if not self.is_test:
                    new_entities = self.update_entities(ini_entities, pre2cur_index)

            # split long entities
            if (not self.is_test) and 'split_entity' in operators:
                new_entities = self.split_entity_on_marks(new_entities)

            new_samples.append({
                self.text: new_text,
                self.entities: new_entities
            })

        # split Long sentence
        if 'split_text' in operators:
            new_samples = self.split_sentence_update_entities(new_samples, max_len)

        # check generated data
        if not self.check_generated_samples(new_samples):
            raise Exception('The generation failed with some bad cases.')

        # save processed data
        save_json_data(new_samples, save_path)

#python DataProcessor.py --data_file "Data/CMeEE_train.json" --save_file_name "Data_clean/new.json"


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--is_test', action='store_true', default=False,
                        help='indicate whether is a test file, without any entities.')
    parser.add_argument('--max_sen_len', type=int, default=128,
                        help='The maximum length of the input text, input text will be split if longer.')
    parser.add_argument('--data_file', type=str, required=True,
                        help='Data file to be processed.')
    parser.add_argument('--save_file_name', type=str, required=True,
                        help='The file name of the processed file.')
    args = parser.parse_args()

    dp_train = DataProcessor(is_test=args.is_test)
    dp_train.preprocess(data_path=args.data_file,
                        save_path=args.save_file_name,
                        max_len=args.max_sen_len)












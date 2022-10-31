import numpy as np
import json
import os

from acccalc import __get_class_span_dict__
import torch.utils.data as data

class FewRelDataset(data.Dataset):
    """
    FewRel Dataset
    """
    def __init__(self, path, tokenizer, N, K, na_prob):
        if not os.path.exists(path):
            print("[ERROR] Data file does not exist! ", path)
            assert(0)
        self.json_data = json.load(open(path))
        self.classes = list(self.json_data.keys())
        self.N = N
        self.K = K
        self.na_rate = na_prob
        self.tokenizer = tokenizer

    def __getraw__(self, raw_tokens, head=None, tail=None, is_query=False):
        tokens = ['[CLS']
        indexed_mask = [0]
        cur_pos = 0
        for token in raw_tokens:
            token = token.lower()
            if not is_query and cur_pos == head[0]:
                tokens += ['[unused1]']
                indexed_mask += [0]
            if not is_query and cur_pos == tail[0]:
                tokens += ['[unused2]']
                indexed_mask += [0]
            temp = self.tokenizer.tokenize(token)
            if len(temp) == 0:
                tokens += ['[unused0]']
                indexed_mask += [1]
            else:
                tokens += temp
                indexed_mask += [1] + [0] * (len(temp) - 1)
            if not is_query and cur_pos == head[-1]:
                tokens += ['[unused3]']
                indexed_mask += [0]
            if not is_query and cur_pos == tail[-1]:
                tokens += ['[unused4]']
                indexed_mask += [0]
            cur_pos += 1
        tokens += ['[SEP]']
        indexed_mask += [0]
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokens)
        assert len(indexed_tokens) == len(indexed_mask)
        return indexed_tokens, indexed_mask

    def __getitem__(self, index):
        max_fusion_len = 0
        target_classes = np.random.choice(self.classes, self.N, False)
        cur_prob = np.random.random()
        if cur_prob < self.na_rate:
            query_class = np.random.choice(list(set(self.classes) - set(target_classes)), 1, False)
        else:
            query_class = np.random.choice(target_classes, 1, False)
        fusion = {'token':[], 'mask':[], 'indexed_mask': [], 'seg': []}
        support = {'token':[], 'mask':[], 'indexed_mask': [], 'tag': []}
        query = {'raw_token':[], 'token':[], 'mask':[], 'indexed_mask': [], 'tag': [], 'rel':0}

        for i, class_name in enumerate(target_classes):
            if query_class == class_name:
                sample_num = self.K + 1
            else:
                sample_num = self.K
            indices = np.random.choice(list(range(len(self.json_data[class_name]))), sample_num, False)
            for j in range(len(indices)):
                idx = indices[j]
                raw_token = self.json_data[class_name][idx]['tokens']
                head_idxs = self.json_data[class_name][idx]['h'][2][0]
                tail_idxs = self.json_data[class_name][idx]['t'][2][0]
                tag = [0] * len(raw_token)
                tag[head_idxs[0]:head_idxs[-1] + 1] = [1] * len(head_idxs)
                tag[tail_idxs[0]:tail_idxs[-1] + 1] = [2] * len(tail_idxs)
                assert len(raw_token) == len(tag)
                if j < self.K:
                    indexed_token, indexed_mask  = self.__getraw__(raw_token, head_idxs, tail_idxs)
                    support['token'].append(indexed_token)
                    support['indexed_mask'].append(indexed_mask)
                    support['tag'].append(tag)
                else:
                    query['raw_token'].append(raw_token)
                    indexed_token, indexed_mask  = self.__getraw__(raw_token, is_query=True)
                    query['token'].append(indexed_token)
                    query['indexed_mask'].append(indexed_mask)
                    query['tag'].append(tag)
                    option = [0] * self.N
                    option[i] = 1
                    query['rel'] = option

        if query_class[0] not in target_classes:
            idx = np.random.choice(list(range(len(self.json_data[query_class[0]]))), 1, False)[0]
            raw_token = self.json_data[query_class[0]][idx]['tokens']
            head_idxs = self.json_data[query_class[0]][idx]['h'][2][0]
            tail_idxs = self.json_data[query_class[0]][idx]['t'][2][0]
            tag = [0] * len(raw_token)
            tag[head_idxs[0]:head_idxs[-1] + 1] = [1] * len(head_idxs)
            tag[tail_idxs[0]:tail_idxs[-1] + 1] = [2] * len(tail_idxs)
            if len(raw_token) != len(tag):
                print(raw_token)
                print(tag)
                print(head_idxs)
                print(tail_idxs)

            query['raw_token'].append(raw_token)
            indexed_token, indexed_mask  = self.__getraw__(raw_token, is_query=True)
            query['token'].append(indexed_token)
            query['indexed_mask'].append(indexed_mask)
            option = [0] * self.N
            query['rel'] = option

        q_token = query['token'][0][1:]
        for i in range(len(support['token'])):
            s_len = len(support['token'][i])
            fusion['token'].append(support['token'][i] + q_token)
            token = fusion['token'][i]
            if len(token) > max_fusion_len:
                max_fusion_len = len(token)
            mask = [1] * len(token)
            fusion['mask'].append(mask)
            fusion['indexed_mask'].append(support['indexed_mask'][i] + query['indexed_mask'][0][1:])
            seg = [1] * len(token)
            seg[:s_len] = [0] * s_len
            fusion['seg'].append(seg)

        pad_id = self.tokenizer.convert_tokens_to_ids('[PAD]')
        for indexed_token, mask, seg in zip(fusion['token'], fusion['mask'], fusion['seg']):
            while len(indexed_token) < max_fusion_len:
                indexed_token.append(pad_id)
                mask.append(0)
                seg.append(1)

        data = {'text':{'fusion': fusion,
                        'support': support, 
                        'query': query},
                'relations':{'raw_token': query['raw_token'], 'tags' : query['tag'], 'type' : query['rel']}}
        return data
    
    def __len__(self):
        return 1000000000


class WebnlgDataset(data.Dataset):
    """
    Webnlg Dataset
    """
    def __init__(self, path, tokenizer, N, K, na_prob):
        if not os.path.exists(path):
            print("[ERROR] Data file does not exist! ", path)
            assert(0)
        self.json_datas = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                self.json_datas.append(json.loads(line))
        self.N = N
        self.K = K
        self.na_rate = na_prob
        self.tokenizer = tokenizer

    def __getraw__(self, raw_tokens, head=None, tail=None, is_query=False):
        tokens = ['[CLS']
        indexed_mask = [0]
        cur_pos = 0
        for token in raw_tokens:
            token = token.lower()
            if not is_query and cur_pos == head[0]:
                tokens += ['[unused1]']
                indexed_mask += [0]
            if not is_query and cur_pos == tail[0]:
                tokens += ['[unused2]']
                indexed_mask += [0]
            temp = self.tokenizer.tokenize(token)
            if len(temp) == 0:
                tokens += ['[unused0]']
                indexed_mask += [1]
            else:
                tokens += temp
                indexed_mask += [1] + [0] * (len(temp) - 1)
            if not is_query and cur_pos == head[-1]:
                tokens += ['[unused3]']
                indexed_mask += [0]
            if not is_query and cur_pos == tail[-1]:
                tokens += ['[unused4]']
                indexed_mask += [0]
            cur_pos += 1
        tokens += ['[SEP]']
        indexed_mask += [0]
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokens)
        assert len(indexed_tokens) == len(indexed_mask)
        return indexed_tokens, indexed_mask

    def __satisfy__(self, target):
        is_satisfy = True
        if len(target) == self.N:
            for k in target:
                if target[k] != self.K:
                    is_satisfy = False
                    break
        else:
            is_satisfy = False

        return is_satisfy
    
    def __random_insert__(self, map, rel):
        idxs = []
        for i in range(len(map)):
            if map[i] == '':
                idxs.append(i)
        idx = np.random.choice(idxs, 1, False)[0]
        map[idx] = rel

    def __getitem__(self, index):
        max_fusion_len = 0
        fusion = {'token':[], 'mask':[], 'indexed_mask': [], 'seg': []}
        support = {'token':[0] * self.N * self.K, 
                   'mask':[0]  * self.N * self.K, 
                   'indexed_mask': [0]  * self.N * self.K, 
                   'tag': [0]  * self.N * self.K}
        query = {'raw_token':[], 'token':[], 'mask':[], 'indexed_mask': [], 'tag': [], 'rel':0}

        idxs = list(range(len(self.json_datas)))
        target_classes = {}
        rel_map = [''] * self.N
        used_idxs = []
        while len(target_classes) == 0 or len(target_classes) > self.N:
            target_classes = {}
            query_idx = np.random.choice(idxs, 1, False)[0]
            query_sent = self.json_datas[query_idx]
            
            skip = False
            for relation in query_sent['relations']:
                rtext = relation['rtext']
                if not rtext in target_classes:
                    cur_prob = np.random.random()
                    if cur_prob < self.na_rate:
                        skip = True
                        pass
                    else:
                        target_classes[rtext] = 0
                        self.__random_insert__(rel_map, rtext)
            if skip and len(target_classes) <= self.N:
                break
        used_idxs.append(query_idx)

        raw_token = query_sent['sentext']
        query['raw_token'].append(raw_token)
        indexed_token, indexed_mask  = self.__getraw__(raw_token, is_query=True)
        query['token'].append(indexed_token)
        query['indexed_mask'].append(indexed_mask)
        option = [0] * self.N

        for relation in query_sent['relations']:
            head = tuple(relation['h'][0] + relation['h'][1])
            tail = tuple(relation['t'][0] + relation['t'][1])
            head_idxs = relation['h'][1]
            tail_idxs = relation['t'][1]
            tag = [0] * len(raw_token)
            tag[head_idxs[0]:head_idxs[-1] + 1] = [1] * (head_idxs[-1] - head_idxs[0] + 1)
            tag[tail_idxs[0]:tail_idxs[-1] + 1] = [2] * (tail_idxs[-1] - tail_idxs[0] + 1)
            
            try:
                i = rel_map.index(relation['rtext'])
                query['tag'].append(tag)
                
                option[i] = 1
            except:
                continue
        query['rel'] = option

        while not self.__satisfy__(target_classes):
            support_idx = np.random.choice(idxs, 1, False)[0]
            if support_idx in used_idxs:
                continue
            else:
                used_idxs.append(support_idx)

            support_sent = self.json_datas[support_idx]
            for relation in support_sent['relations']:
                rtext = relation['rtext']
                
                if not rtext in target_classes:
                    if len(target_classes) < self.N:
                        target_classes[rtext] = 0
                        self.__random_insert__(rel_map, rtext)
                    else:
                        continue
                if target_classes[rtext] < self.K:
                    target_classes[rtext] += 1
                else:
                    continue
                base_idx = rel_map.index(rtext)
                raw_token = support_sent['sentext']
                head_idxs = relation['h'][1]
                tail_idxs = relation['t'][1]
                tag = [0] * len(raw_token)
                head = tuple(relation['h'][0] + relation['h'][1])
                tail = tuple(relation['t'][0] + relation['t'][1])

                tag[head_idxs[0]:head_idxs[-1] + 1] = [1] * (head_idxs[-1] - head_idxs[0] + 1)
                tag[tail_idxs[0]:tail_idxs[-1] + 1] = [2] * (tail_idxs[-1] - tail_idxs[0] + 1)
                # if not 0 in tag or not 1 in tag or not 2 in tag:
                #     print(tag)
                #     print(head)
                #     print(tail)
                assert len(raw_token) == len(tag)
                indexed_token, indexed_mask = self.__getraw__(raw_token, head_idxs, tail_idxs)
                support['token'][base_idx*self.K+target_classes[rtext]-1] = indexed_token
                support['indexed_mask'][base_idx*self.K+target_classes[rtext]-1] = indexed_mask
                support['tag'][base_idx*self.K+target_classes[rtext]-1] = tag

        q_token = query['token'][0][1:]
        for i in range(len(support['token'])):
            s_len = len(support['token'][i])
            fusion['token'].append(support['token'][i] + q_token)
            token = fusion['token'][i]
            if len(token) > max_fusion_len:
                max_fusion_len = len(token)
            mask = [1] * len(token)
            fusion['mask'].append(mask)
            fusion['indexed_mask'].append(support['indexed_mask'][i] + query['indexed_mask'][0][1:])
            seg = [1] * len(token)
            seg[:s_len] = [0] * s_len
            fusion['seg'].append(seg)

        pad_id = self.tokenizer.convert_tokens_to_ids('[PAD]')
        for indexed_token, mask, seg in zip(fusion['token'], fusion['mask'], fusion['seg']):
            while len(indexed_token) < max_fusion_len:
                indexed_token.append(pad_id)
                mask.append(0)
                seg.append(1)

        data = {'text':{'fusion': fusion,
                        'support': support, 
                        'query': query},
                'relations':{'raw_token': query['raw_token'], 'tags' : query['tag'], 'type' : query['rel']}}
        return data
    
    def __len__(self):
        return 1000000000

class DataManager:
    def __init__(self, tokenizer, N, K, dataset='fewrel', na_prob=0.0):
        self.data = {}
        if dataset != 'webnlg':
            self.data['train'] = FewRelDataset('./datasets/'+dataset+'/train.txt', tokenizer, N, K, na_prob)
            self.data['dev'] = FewRelDataset('./datasets/'+dataset+'/dev.txt', tokenizer, N, K, na_prob)
            self.data['test'] = FewRelDataset('./datasets/'+dataset+'/test.txt', tokenizer, N, K, na_prob)
        else:
            self.data['train'] = FewRelDataset('./datasets/'+dataset+'/train.txt', tokenizer, N, K, na_prob)
            self.data['dev'] = WebnlgDataset('./datasets/'+dataset+'/dev.txt', tokenizer, N, K, na_prob)
            self.data['test'] = WebnlgDataset('./datasets/'+dataset+'/test.txt', tokenizer, N, K, na_prob)


def getraw(raw_tokens, head, tail, tokenizer):
    tokens = []
    indexed_mask = []
    cur_pos = 0
    for token in raw_tokens:
        token = token.lower()
        if cur_pos == head[0]:
            tokens += ['[unused1]']
            indexed_mask += [0]
        if cur_pos == tail[0]:
            tokens += ['[unused2]']
            indexed_mask += [0]
        temp = tokenizer.tokenize(token)
        if len(temp) == 0:
            tokens += ['[unused0]']
            indexed_mask += [1]
        else:
            tokens += temp
            indexed_mask += [1] + [0] * (len(temp) - 1)
        if cur_pos == head[-1]:
            tokens += ['[unused3]']
            indexed_mask += [0]
        if cur_pos == tail[-1]:
            tokens += ['[unused4]']
            indexed_mask += [0]
        cur_pos += 1
    tokens += ['[SEP]']
    indexed_mask += [0]
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokens)
    assert len(indexed_tokens) == len(indexed_mask)
    return indexed_tokens, indexed_mask

def recombine_fusion(actions, actprobs, support_token, query, tokenizer, cnt, training):
    fusion = {'token':[], 'mask':[], 'seg': [], 'new_rel': []}
    query_raw_token = query['raw_token'][0]
    if training:
        query_tag = query['tag'][cnt]
        label_class_span = __get_class_span_dict__(query_tag)
        pred_class_span = __get_class_span_dict__(actions)

        pred_class_span_with_prob = {}
        for label in label_class_span:
            pred_class_span_with_prob[label] = []
            for span in pred_class_span.get(label, []):
                span_avg_prob = np.mean([item.cpu().detach().item() for item in actprobs[span[0]:span[1]]])
                pred_class_span_with_prob[label].append((span_avg_prob, span))
            pred_class_span_with_prob[label].sort(key=lambda item:item[0], reverse=True)
        total_spans_list = []
        total_spans_label = []
        total_spans_list.append((label_class_span[1][0], label_class_span[2][0]))
        total_spans_label.append(1)
        cnt = 0
        for i in range(len(pred_class_span_with_prob[2])):
            if label_class_span[2][0] != pred_class_span_with_prob[2][i][1]:
                total_spans_list.append((label_class_span[1][0], pred_class_span_with_prob[2][i][1]))
                total_spans_label.append(0)
                cnt += 1
            if cnt >= 2:
                break
        cnt = 0
        for i in range(len(pred_class_span_with_prob[1])):
            if label_class_span[1][0] != pred_class_span_with_prob[1][i][1]:
                total_spans_list.append((pred_class_span_with_prob[1][i][1], label_class_span[2][0]))
                total_spans_label.append(0)
                cnt += 1
            if cnt >= 2:
                break
        list_index = [i for i in range(len(total_spans_label))]
        np.random.shuffle(list_index)
        total_spans_list = [total_spans_list[i] for i in list_index]
        total_spans_label = [total_spans_label[i] for i in list_index]
    else:
        pred_class_span = __get_class_span_dict__(actions)
        total_spans_list = []
        if len(pred_class_span.get(1, [])) == 0:
            pred_class_span[1] = [(-1,-1)]
        if len(pred_class_span.get(2, [])) == 0:
            pred_class_span[2] = [(-1,-1)]
        for span1 in pred_class_span[1]:
            for span2 in pred_class_span[2]:
                total_spans_list.append((span1, span2))

    max_fusion_len = 0
    for j in range(len(total_spans_list)):
        span = total_spans_list[j]
        if training:
            fusion['new_rel'].append(total_spans_label[j])
        q_token, _ = getraw(query_raw_token, span[0], span[1], tokenizer)
        for i in range(len(support_token)):
            s_len = len(support_token[i])
            fusion['token'].append(support_token[i] + q_token)
            token = fusion['token'][j*len(support_token) + i]
            if len(token) > max_fusion_len:
                max_fusion_len = len(token)
            mask = [1] * len(token)
            fusion['mask'].append(mask)
            seg = [1] * len(token)
            seg[:s_len] = [0] * s_len
            fusion['seg'].append(seg)
    pad_id = tokenizer.convert_tokens_to_ids('[PAD]')
    for indexed_token, mask, seg in zip(fusion['token'], fusion['mask'], fusion['seg']):
        while len(indexed_token) < max_fusion_len:
            indexed_token.append(pad_id)
            mask.append(0)
            seg.append(1)
    return fusion
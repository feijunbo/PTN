import json
import random
import re


train_path = "./nyt10m/nyt10m_train.txt"
eval_path = "./nyt10m/nyt10m_val.txt"
test_path = "./nyt10m/nyt10m_test.txt"

files = [train_path, eval_path, test_path]

text_datas = {}
for file in files:
    with open(file, 'r', encoding='utf-8') as f:
        for line in f:
            json_data = json.loads(line)
            text = json_data['text']
            if text not in text_datas:
                text_datas[text] = 0
            text_datas[text] += 1

json_datas = {}
for file in files:
    with open(file, 'r', encoding='utf-8') as f:
        for line in f:
            json_data = json.loads(line)
            text = json_data['text']
            relation = json_data['relation']
            if text_datas[text] == 1:
                if relation not in json_datas:
                    json_datas[relation] = []
                json_datas[relation].append(json_data)

del json_datas['NA']
keys = list(json_datas.keys())
for key in keys:
    if len(json_datas[key]) < 50:
        del json_datas[key]
    # else:
    #     print(key,":", len(json_datas[key]))

all_keys = list(json_datas.keys())

def get_sentence(sentence, position):
    cvt_pos = [0, 0, 0, 0]
    sentence = re.split('(\W+)', sentence)
    idx = 0
    for i in range(len(sentence)):
        if idx in position:
            cvt_pos[position.index(idx)] = i
        idx += len(sentence[i])
    for i in range(len(sentence))[::-1]:
        if sentence[i] == ' ' or sentence[i] == '':
            sentence.pop(i)
            if i < cvt_pos[0]:
                cvt_pos[0] -= 1
                cvt_pos[1] -= 1
            if i > cvt_pos[0] and i < cvt_pos[1]:
                cvt_pos[1] -= 1
            if i < cvt_pos[2]:
                cvt_pos[2] -= 1
                cvt_pos[3] -= 1
            if i > cvt_pos[2] and i < cvt_pos[3]:
                cvt_pos[3] -= 1
            continue
        sentence[i] = sentence[i].strip()
    return sentence, cvt_pos

def get_entity(entity):
    entity = re.split('(\W+)', entity)
    for i in range(len(entity))[::-1]:
        if entity[i] == ' ' or entity[i] == '':
            entity.pop(i)
            continue
        entity[i] = entity[i].strip()
    return entity

standard_jsons = {}
for key in all_keys:
    datas = json_datas[key]
    random.shuffle(datas)
    for data in datas:
        position = data['h']['pos'] + data['t']['pos']
        splited_sent, cvt_pos = get_sentence(data['text'], position)
        h_name = data['h']['name']
        t_name = data['t']['name']
        head_len = len(get_entity(h_name))
        tail_len = len(get_entity(t_name))
        head_idx = [i for i in range(cvt_pos[0], cvt_pos[1])]
        tail_idx = [i for i in range(cvt_pos[2], cvt_pos[3])]
        if len(head_idx) == 0 or head_len != (head_idx[-1] - head_idx[0] + 1):
            continue
        if len(tail_idx) == 0 or tail_len != (tail_idx[-1] - tail_idx[0] + 1):
            continue
        if tuple(splited_sent[head_idx[0]:head_idx[-1] + 1]) != tuple(get_entity(h_name)):
            print(splited_sent[head_idx[0]:head_idx[-1] + 1])
            print(get_entity(h_name))
            print(splited_sent)
        if tuple(splited_sent[tail_idx[0]:tail_idx[-1] + 1]) != tuple(get_entity(t_name)):
            print(splited_sent[tail_idx[0]:tail_idx[-1] + 1])
            print(get_entity(t_name))
            print(splited_sent)
        standard_json = {'tokens':splited_sent, 'h':[h_name, data['h']['id'], [head_idx]], 't':[t_name, data['t']['id'], [tail_idx]]}
        if key not in standard_jsons:
            standard_jsons[key] = []
        standard_jsons[key].append(standard_json)
        if len(standard_jsons[key]) == 50:
            break
with open('./test.txt', 'w', encoding='utf-8') as f:
    json.dump(standard_jsons, f, ensure_ascii=False)


import os
import re
import json
from xml.dom import minidom
import random
# from sys import argv

# script,first = argv
# random.seed(first)

def list_all_files(rootdir):
    _files = []

    list_file = os.listdir(rootdir)
    
    for i in range(0,len(list_file)):

        path = os.path.join(rootdir,list_file[i])

        if os.path.isdir(path):
            _files.extend(list_all_files(path))
        if os.path.isfile(path):
             _files.append(path)

    return _files

def get_entity(entity):
    
    for i in range(len(entity))[::-1]:
        if entity[i][0] == '(' and entity[i][-1] == ')':
            entity.pop(i)
            continue
    entity = re.split('(\W+)', ' '.join(entity))
    for i in range(len(entity))[::-1]:
        if entity[i] == ' ' or entity[i] == '':
            entity.pop(i)
            continue
        entity[i] = entity[i].strip()
    return entity

def get_sentence(sentence):
    sentence = re.split('(\W+)', sentence)
    for i in range(len(sentence))[::-1]:
        if sentence[i] == ' ' or sentence[i] == '':
            sentence.pop(i)
            continue
        sentence[i] = sentence[i].strip()
    return sentence

src_dir = './release_v3.0/en/train'
files = list_all_files(src_dir)
src_dir = './release_v3.0/en/dev'
files.extend(list_all_files(src_dir))

relations = {}
all_relations = {}
json_datas = []
for file in files:
    xmldoc = minidom.parse(file)
    xmldoc = xmldoc.documentElement
    entries = xmldoc.getElementsByTagName('entry')
    for entry in entries:
        category = entry.getAttribute('category')
        if not category in relations:
            relations[category] = {}
        mtriples = entry.getElementsByTagName('mtriple')
        triples = []
        entities = []
        for m in mtriples:
            ms = m.firstChild.nodeValue
            ms = ms.strip().split(' | ')
            subject = tuple(get_entity(ms[0].split('_')))
            object = tuple(get_entity(ms[2].split('_')))
            relation = ms[1]
            triples.append((relation, subject, object))
            if not subject in entities:
                entities.append(subject)
            if not object in entities:
                entities.append(object)

        lexes = entry.getElementsByTagName('lex')

        for l in lexes:
            ls = l.firstChild.nodeValue
            sentence = get_sentence(ls)
            positions = []
            isValid = True
            for i in range(len(entities)):
                cnt = 0
                for j in range(len(sentence) - len(entities[i])):
                    if tuple(sentence[j:j+len(entities[i])]) == entities[i]:
                        cnt += 1
                        if cnt > 1:
                            break
                        positions.append([j,j+len(entities[i]) - 1])
                if cnt != 1:
                    isValid = False
                    break

            if isValid:
                json_data = {}
                json_data['sentext'] = sentence
                json_data['relations'] = []
                for t in triples:
                    triple = {}
                    triple['rtext'] = t[0]
                    triple['h'] = [list(t[1]), positions[entities.index(t[1])]]
                    triple['t'] = [list(t[2]), positions[entities.index(t[2])]]
                    head = triple['h'][1]
                    tail = triple['t'][1]
                    if head[0] >= tail[0] and head[0] <= tail[1]:
                        continue
                    if head[1] >= tail[0] and head[1] <= tail[1]:
                        continue
                    if tail[1] >= head[0] and tail[1] <= head[1]:
                        continue
                    if tail[0] >= head[0] and tail[0] <= head[1]:
                        continue
                    json_data['relations'].append(triple)
                    if not t[0] in relations[category]:
                        relations[category][t[0]] = 0
                    relations[category][t[0]] += 1
                    if not t[0] in all_relations:
                        all_relations[t[0]] = 0
                    all_relations[t[0]] += 1
                json_datas.append(json_data)

sorted(all_relations.items(), key=lambda d: d[1])
cnt = 0
r_cnt50 = []
for s in all_relations:
    if all_relations[s] > 150:
        cnt += 1
        r_cnt50.append(s)

for s in relations:
    keys = list(relations[s].keys())
    for t in keys:
        if not t in r_cnt50:
            relations[s].pop(t)

for j in range(len(json_datas))[::-1]:
    relations = json_datas[j]['relations']
    for i in range(len(relations))[::-1]:
        if not relations[i]['rtext'] in r_cnt50:
            relations.pop(i)
    if len(relations) == 0:
        json_datas.pop(j)
        continue

# with open('./all.json', 'w', encoding='utf-8') as f:
#     for json_data in json_datas:
#         f.write(json.dumps(json_data, ensure_ascii=False)+'\n')

# json_datas = []
# with open('./all.json', 'r', encoding='utf-8') as f:
#     for line in f:
#         json_datas.append(json.loads(line))

idx_list = [i for i in range(len(json_datas))]
random.shuffle(idx_list)

test_keys = []
dev_keys = []

i = 0
while len(test_keys) < 16:
    json_data = json_datas[idx_list[i]]
    relations = json_data['relations']
    for relation in relations:
        rtext = relation['rtext']
        if not rtext in test_keys:
            test_keys.append(rtext)
    i += 1

while len(dev_keys) < 15:
    json_data = json_datas[idx_list[i]]
    relations = json_data['relations']
    for relation in relations:
        rtext = relation['rtext']
        if not rtext in dev_keys and not rtext in test_keys:
            dev_keys.append(rtext)
    i += 1

with open('./test.txt', 'w', encoding='utf-8') as testin, \
     open('./dev.txt', 'w', encoding='utf-8') as devin:
    for json_data in json_datas:
        relations = json_data['relations']
        # train_json = {'sentext': json_data['sentext'], 'relations': []}
        dev_json = {'sentext': json_data['sentext'], 'relations': []}
        test_json = {'sentext': json_data['sentext'], 'relations': []}
        for relation in relations:
            rtext = relation['rtext']
            if rtext in test_keys:
                test_json['relations'].append(relation)
            elif rtext in dev_keys:
                dev_json['relations'].append(relation)
        if len(test_json['relations']) != 0:
            testin.write(json.dumps(test_json, ensure_ascii=False) + '\n')
        if len(dev_json['relations']) != 0:
            devin.write(json.dumps(dev_json, ensure_ascii=False) + '\n')


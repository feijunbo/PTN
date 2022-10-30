import json
import random

train_path = "./train_wiki.json"
eval_path = "./val_wiki.json"

eval_json_data = json.load(open(eval_path))
train_json_data = json.load(open(train_path))


te_keys = list(eval_json_data.keys())
tr_keys = list(train_json_data.keys())
random.shuffle(te_keys)
random.shuffle(tr_keys)
test_keys = te_keys[:15]

dev_keys = te_keys[15:] + tr_keys[-14:]
train_keys = tr_keys[:-14]

with open('./dev.txt', 'w', encoding='utf-8') as f:
    json_data = {}
    for key in dev_keys:
        try:
            json_data[key] = eval_json_data[key]
        except:
            json_data[key] = train_json_data[key]
    json.dump(json_data, f, ensure_ascii=False)

with open('./test.txt', 'w', encoding='utf-8') as f:
    json_data = {}
    for key in test_keys:
        try:
            json_data[key] = eval_json_data[key]
        except:
            json_data[key] = train_json_data[key]
    json.dump(json_data, f, ensure_ascii=False)

with open('./train.txt', 'w', encoding='utf-8') as f:
    json_data = {}
    for key in train_keys:
        json_data[key] = train_json_data[key]
    json.dump(json_data, f, ensure_ascii=False)
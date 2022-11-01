import time
import torch
import queue

from transformers import AdamW, get_linear_schedule_with_warmup
from acccalc import calc_acc, calc_ent_acc, calc_post_acc, calc_rel_acc, error_analysis, error_detail, rule_actions, update_statictis
from optimize import optimize_round
from acccalc import __get_class_span_dict__

def recombine_action(rel_action, ent_action, tri_action):
    j = 0
    for i in range(len(rel_action)):
        if rel_action[i] > 0:
            pred_class_span = __get_class_span_dict__(ent_action[j])
            if len(pred_class_span.get(1,[])) == 0:
                pred_class_span[1] = [(-1,-1)]
            if len(pred_class_span.get(2, [])) == 0:
                pred_class_span[2] = [(-1,-1)]
            span1s = []
            span2s = []
            cnt = 0
            for span1 in pred_class_span[1]:
                for span2 in pred_class_span[2]:
                    if tri_action[j][cnt] == 1:
                        span1s.append(span1)
                        span2s.append(span2)
                    cnt += 1
            span1s = list(set(span1s))
            span2s = list(set(span2s))
            ent_action[j] = [0] * len(ent_action[j])
            for span in span1s:
                if span != (-1,-1):
                    ent_action[j][span[0]:span[1]] = [1] * (span[1] - span[0])
            for span in span2s:
                if span != (-1,-1):
                    ent_action[j][span[0]:span[1]] = [2] * (span[1] - span[0])
            j += 1

def workProcess(model, datas, mode, alpha, beta, gamma):
    acc, cnt, tot = 0, 0, 0
    new_acc, new_cnt, new_tot = 0, 0, 0
    post_acc, post_cnt, post_tot = 0, 0, 0
    rel_acc, rel_cnt, rel_tot = 0, 0, 0
    ent_acc, ent_cnt, ent_tot = 0, 0, 0
    loss = .0
    error_statictis = {}
    error_details = []
    for data in datas:
        preoptions, preactions = rule_actions(data['relations'])
        if "test" not in mode:
            rel_action, rel_actprob, ent_action, ent_actprob, tri_action, tri_actprob = \
                    model(mode, data['text'], preoptions, preactions)
        else:
            with torch.no_grad():
                rel_action, rel_actprob, ent_action, ent_actprob, tri_action, tri_actprob = \
                        model(mode, data['text'])

        acc1, tot1, cnt1 = calc_acc(rel_action, ent_action, \
                data['relations'], mode)
        acc += acc1
        tot += tot1
        cnt += cnt1
        if "test" in mode:
            recombine_action(rel_action, ent_action, tri_action)
        acc1, tot1, cnt1 = calc_acc(rel_action, ent_action, \
                data['relations'], mode)
        new_acc += acc1
        new_tot += tot1
        new_cnt += cnt1
        acc1, tot1, cnt1 = calc_post_acc(rel_action, ent_action, \
                data['relations'], ent_actprob, mode)
        post_acc += acc1
        post_tot += tot1
        post_cnt += cnt1
        acc1, tot1, cnt1 = calc_rel_acc(rel_action, ent_action, \
                data['relations'], mode)
        rel_acc += acc1
        rel_tot += tot1
        rel_cnt += cnt1
        acc1, tot1, cnt1 = calc_ent_acc(rel_action, ent_action, \
                data['relations'], mode)
        ent_acc += acc1
        ent_tot += tot1
        ent_cnt += cnt1
        if "test" in mode:
            statictis = error_analysis(rel_action, ent_action, \
                    data['relations'], mode)
            update_statictis(statictis, error_statictis)
            detail = error_detail(rel_action, ent_action, \
                    data['relations'], ent_actprob)
            error_details.append(detail)
        if "test" not in mode:
            loss += optimize_round(rel_action, rel_actprob, ent_action,\
                    ent_actprob, tri_action, tri_actprob, mode, alpha, beta, gamma)
    return acc, cnt, tot, rel_acc, rel_cnt, rel_tot, ent_acc, ent_cnt, ent_tot, post_acc, post_tot, post_cnt, new_acc, new_cnt, new_tot, error_statictis, error_details, loss / len(datas)

def worker(model, rank, dataQueue, resultQueue, freeProcess, lock, flock, lr, alpha, beta, gamma):
    parameters_to_optimize = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    parameters_to_optimize = [
        {'params': [p for n, p in parameters_to_optimize 
            if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in parameters_to_optimize
            if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    # optimizer = optim.Adam(parameters_to_optimize, lr=lr)
    optimizer = AdamW(parameters_to_optimize, lr=lr, correct_bias=False)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=300, num_training_steps=20000) 
    print("Process ", rank, " start service.")
    flock.acquire()
    freeProcess.value += 1
    flock.release()
    while True:
        datas, mode, dataID = dataQueue.get()
        flock.acquire()
        freeProcess.value -= 1
        flock.release()
        model.zero_grad()
        acc, cnt, tot, rel_acc, rel_cnt, rel_tot, ent_acc, ent_cnt, ent_tot, post_acc, post_tot, post_cnt, new_acc, new_cnt, new_tot, error_statictis, error_details, loss = workProcess(model, datas, mode, alpha, beta, gamma)
        resultQueue.put((acc, cnt, tot, rel_acc, rel_cnt, rel_tot, ent_acc, ent_cnt, ent_tot, post_acc, post_tot, post_cnt, new_acc, new_cnt, new_tot, error_statictis, error_details, dataID, rank, loss))
        if not "test" in mode:
            lock.acquire()
            optimizer.step()
            scheduler.step()
            lock.release()
        flock.acquire()
        freeProcess.value += 1
        flock.release()

def train(dataID, model, datas, mode, dataQueue, resultQueue, freeProcess, lock, numprocess):
    dataPerProcess = len(datas) // numprocess
    while freeProcess.value != numprocess:
        pass

    acc, cnt, tot = 0, 0, 0
    new_acc, new_cnt, new_tot = 0, 0, 0
    post_acc, post_tot, post_cnt = 0, 0, 0
    rel_acc, rel_cnt, rel_tot = 0, 0, 0
    ent_acc, ent_cnt, ent_tot = 0, 0, 0
    error_statictis = {}
    error_details = []
    loss = .0
    for r in range(numprocess):
        endPos = ((r+1)*dataPerProcess if r+1 != numprocess else len(datas))
        data = datas[r*dataPerProcess: endPos]
        dataQueue.put((data, mode, dataID))
    lock.acquire()
    try:
        for r in range(numprocess):
            while True:
                item = resultQueue.get()
                if item[17] == dataID:
                    break
                else:
                    print ("receive wrong dataID: ", item[17], "from process ", item[18])
            acc += item[0]
            cnt += item[1]
            tot += item[2]
            rel_acc += item[3]
            rel_cnt += item[4]
            rel_tot += item[5]
            ent_acc += item[6]
            ent_cnt += item[7]
            ent_tot += item[8]
            post_acc += item[9]
            post_tot += item[10]
            post_cnt += item[11]
            new_acc += item[12]
            new_cnt += item[13]
            new_tot += item[14]
            statictis = item[15]
            details = item[16]
            update_statictis(statictis, error_statictis)
            error_details.extend(details)
            loss += item[19]
    except queue.Empty:
        print("The result of some process missed...")
        print(freeProcess.value)
        lock.release()
        time.sleep(2)
        print(freeProcess.value)
        while True:
            pass

    lock.release()
    # if dataID > 0 and dataID % 200 == 0:
    #     print('=========',dataID,'=========')
    #     print('loss    :', loss / numprocess)
        # print('joint   :', acc, cnt, tot)
        # print('new     :', new_acc, new_cnt, new_tot)
        # print('post    :', post_acc, post_cnt, post_tot)
        # print('relation:', rel_acc, rel_cnt, rel_tot)
        # print('entity  :', ent_acc, ent_cnt, ent_tot)

    return (acc, cnt, tot, rel_acc, rel_cnt, rel_tot, ent_acc, ent_cnt, ent_tot, post_acc, post_tot, post_cnt, new_acc, new_cnt, new_tot, error_statictis, error_details)

def test(dataID, model, datas, mode, dataQueue, resultQueue, freeProcess, lock, numprocess):
    testmode = mode + ["test"]
    if dataID < -2:
        print(testmode)
    return train(-dataID-1, model, datas, testmode, dataQueue, resultQueue, freeProcess, lock, numprocess)


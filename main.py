import json
import random, sys, time, os
import torch
import numpy as np
from transformers import BertModel, BertTokenizer
from datamanager import DataManager
from model import Model
from options import Parser
from train import train, test, worker
import torch.multiprocessing as mp
from acccalc import calcF1, update_statictis
import pandas as pd

def work(mode, train_data, test_data, dev_data, model, args, epoch, best=0):
    bestF1 = best
    devF1s = []
    devnF1s = []
    devpF1s = []
    devEntF1s = []
    devRelF1s = []
    testF1s = []
    testnF1s = []
    testpF1s = []
    testEntF1s = []
    testRelF1s = []
    if args.debug:
        epoch = 2
        
    for e in range(epoch):
        print("training epoch ", e)
        if args.debug:
            batchcnt = 200
        else:
            batchcnt = args.train_iter
        acc, cnt, tot = 0, 0, 0
        new_acc, new_cnt, new_tot = 0, 0, 0
        post_acc, post_tot, post_cnt = 0, 0, 0
        rel_acc, rel_cnt, rel_tot = 0, 0, 0
        ent_acc, ent_cnt, ent_tot = 0, 0, 0
        start = time.time()
        model.train()
        for b in range(batchcnt):
            data = []
            for i in range(b * args.batchsize, (b+1) * args.batchsize):
                data.append(train_data[i])
            acc_, cnt_, tot_, rel_acc_, rel_cnt_, rel_tot_, ent_acc_, ent_cnt_, ent_tot_, post_acc_, post_tot_, post_cnt_, new_acc_, new_cnt_, new_tot_, statictis, details = train(b, model, data, \
                    mode, dataQueue, resultQueue, freeProcess, lock, args.numprocess)
            acc += acc_
            cnt += cnt_ 
            tot += tot_
            new_acc += new_acc_
            new_cnt += new_cnt_ 
            new_tot += new_tot_
            post_acc += post_acc_
            post_cnt += post_cnt_
            post_tot += post_tot_
            rel_acc += rel_acc_
            rel_cnt += rel_cnt_ 
            rel_tot += rel_tot_
            ent_acc += ent_acc_
            ent_cnt += ent_cnt_ 
            ent_tot += ent_tot_
            # trainF1 = calcF1(acc, cnt, tot)
            # trainnF1 = calcF1(new_acc, new_cnt, new_tot)
            # trainpF1 = calcF1(post_acc, post_cnt, post_tot)
            # trainRelF1 = calcF1(rel_acc, rel_cnt, rel_tot)
            # trainEntF1 = calcF1(ent_acc, ent_cnt, ent_tot)
            # if (b + 1) % args.print_per_batch == 0:
                # print("    batch ", b, ": F1:", trainF1, ", new F1:", trainnF1, ", post F1:", trainpF1, ", Rel F1:", trainRelF1, ", Ent F1:", trainEntF1, "    time:", (time.time()-start))
                # acc, cnt, tot = 0, 0, 0
                # rel_acc, rel_cnt, rel_tot = 0, 0, 0
                # ent_acc, ent_cnt, ent_tot = 0, 0, 0
                # start = time.time()
        model.eval()
        if args.debug:
            batchcnt = 200
        else:
            batchcnt = args.dev_iter
        acc, cnt, tot = 0, 0, 0
        new_acc, new_cnt, new_tot = 0, 0, 0
        post_acc, post_tot, post_cnt = 0, 0, 0
        rel_acc, rel_cnt, rel_tot = 0, 0, 0
        ent_acc, ent_cnt, ent_tot = 0, 0, 0
        dev_error_statictis = {}
        dev_error_details = []
        for b in range(batchcnt):
            # data = dev_data[b * args.batchsize_test : (b+1) * args.batchsize_test]
            data = []
            for i in range(b * args.batchsize_test, (b+1) * args.batchsize_test):
                data.append(dev_data[i])
            acc_, cnt_, tot_, rel_acc_, rel_cnt_, rel_tot_, ent_acc_, ent_cnt_, ent_tot_, post_acc_, post_tot_, post_cnt_, new_acc_, new_cnt_, new_tot_, statictis, details = test(b, model, data, mode, \
                    dataQueue, resultQueue, freeProcess, lock, args.numprocess)
            acc += acc_
            cnt += cnt_ 
            tot += tot_
            new_acc += new_acc_
            new_cnt += new_cnt_ 
            new_tot += new_tot_
            post_acc += post_acc_
            post_cnt += post_cnt_
            post_tot += post_tot_
            rel_acc += rel_acc_
            rel_cnt += rel_cnt_ 
            rel_tot += rel_tot_
            ent_acc += ent_acc_
            ent_cnt += ent_cnt_ 
            ent_tot += ent_tot_
            update_statictis(statictis, dev_error_statictis)
            dev_error_details.extend(details)
        devF1 = calcF1(acc, cnt, tot)
        devnF1 = calcF1(new_acc, new_cnt, new_tot)
        devpF1 = calcF1(post_acc, post_cnt, post_tot)
        devRelF1 = calcF1(rel_acc, rel_cnt, rel_tot)
        devEntF1 = calcF1(ent_acc, ent_cnt, ent_tot)
        if args.debug:
            batchcnt = 200
        else:
            batchcnt = args.test_iter
        acc, cnt, tot = 0, 0, 0
        new_acc, new_cnt, new_tot = 0, 0, 0
        post_acc, post_tot, post_cnt = 0, 0, 0
        rel_acc, rel_cnt, rel_tot = 0, 0, 0
        ent_acc, ent_cnt, ent_tot = 0, 0, 0
        test_error_details = []
        test_error_statictis = {}
        for b in range(batchcnt):
            # data = test_data[b * args.batchsize_test : (b+1) * args.batchsize_test]
            data = []
            for i in range(b * args.batchsize_test, (b+1) * args.batchsize_test):
                data.append(test_data[i])
            acc_, cnt_, tot_, rel_acc_, rel_cnt_, rel_tot_, ent_acc_, ent_cnt_, ent_tot_, post_acc_, post_tot_, post_cnt_, new_acc_, new_cnt_, new_tot_, statictis, details = test(b, model, data, mode, \
                    dataQueue, resultQueue, freeProcess, lock, args.numprocess)
            acc += acc_
            cnt += cnt_ 
            tot += tot_
            new_acc += new_acc_
            new_cnt += new_cnt_ 
            new_tot += new_tot_
            post_acc += post_acc_
            post_cnt += post_cnt_
            post_tot += post_tot_
            rel_acc += rel_acc_
            rel_cnt += rel_cnt_ 
            rel_tot += rel_tot_
            ent_acc += ent_acc_
            ent_cnt += ent_cnt_ 
            ent_tot += ent_tot_
            update_statictis(statictis, test_error_statictis)
            test_error_details.extend(details)
        testF1 = calcF1(acc, cnt, tot)
        testnF1 = calcF1(new_acc, new_cnt, new_tot)
        testpF1 = calcF1(post_acc, post_cnt, post_tot)
        testRelF1 = calcF1(rel_acc, rel_cnt, rel_tot)
        testEntF1 = calcF1(ent_acc, ent_cnt, ent_tot)
        suffix = "_".join(mode)
        keys = list(dev_error_statictis.keys())
        keys.sort()
        dev_error_statictis = {k:dev_error_statictis[k] for k in keys}
        keys = list(test_error_statictis.keys())
        keys.sort()
        test_error_statictis = {k:test_error_statictis[k] for k in keys}

        f = open("results/"+args.dataset+"_"+str(args.na_prob)+"_"+suffix+"_"+str(args.N)+"_"+str(args.K)+"_"+str(args.seed)+".log", 'a')
        print("epoch ", e, ": dev F1: ", devF1, ", test F1: ", testF1)
        print("epoch ", e, ": new dev F1: ", devnF1, ", new test F1: ", testnF1)
        print("epoch ", e, ": post dev F1: ", devpF1, ", post test F1: ", testpF1)
        print("epoch ", e, ": dev Rel F1: ", devRelF1, ", test Rel F1: ", testRelF1)
        print("epoch ", e, ": dev Ent F1: ", devEntF1, ", test Rel F1: ", testEntF1)
        print("epoch ", e, ": dev error: ", dev_error_statictis, ", test error: ", test_error_statictis)
        f.write("epoch "+ str(e) + ": dev F1: "+ str(devF1) + ", test F1: " + str(testF1) + "\n")
        f.write("epoch "+ str(e) + ": new dev F1: "+ str(devnF1) + ", new test F1: " + str(testnF1) + "\n")
        f.write("epoch "+ str(e) + ": post dev F1: "+ str(devpF1) + ", post test F1: " + str(testpF1) + "\n")
        f.write("epoch "+ str(e) + ": dev Rel F1: "+ str(devRelF1) + ", test Rel F1: " + str(testRelF1) + "\n")
        f.write("epoch "+ str(e) + ": dev Ent F1: "+ str(devEntF1) + ", test Ent F1: " + str(testEntF1) + "\n")
        f.write("epoch "+ str(e) + ": dev error: "+ str(dev_error_statictis) + ", test error: " + str(test_error_statictis) + "\n")
        f.write("\n")
        f.close()
        devF1s.append(devF1)
        devnF1s.append(devnF1)
        devpF1s.append(devpF1)
        devEntF1s.append(devEntF1)
        devRelF1s.append(devRelF1)
        testF1s.append(testF1)
        testnF1s.append(testnF1)
        testpF1s.append(testpF1)
        testEntF1s.append(testEntF1)
        testRelF1s.append(testRelF1)
        devFinalF1 = devpF1
        if bestF1 < devFinalF1:
            f = open("results/"+args.dataset+"_"+str(args.na_prob)+"_"+suffix+"_"+str(args.N)+"_"+str(args.K)+"_"+str(args.seed)+"_dev.detail", 'w')
            for detail in dev_error_details:
                f.write(json.dumps(detail, ensure_ascii=False, indent=2)+'\n')
                f.write('\n')
            f.close()
            f = open("results/"+args.dataset+"_"+str(args.na_prob)+"_"+suffix+"_"+str(args.N)+"_"+str(args.K)+"_"+str(args.seed)+"_test.detail", 'w')
            for detail in test_error_details:
                f.write(json.dumps(detail, ensure_ascii=False, indent=2)+"\n")
                f.write('\n')
            f.close()
            bestF1 = devFinalF1
            state = {'net':model.state_dict(), 'best_F1': bestF1}
            print("best checkpoint!")
            torch.save(state, "checkpoints/best_model_"+args.dataset+"_"+str(args.na_prob)+"_"+suffix+"_"+str(args.N)+"_"+str(args.K)+"_"+str(args.seed)+".pkl")
    df = pd.DataFrame()
    df['devF1'] = devF1s
    df['devnF1'] = devnF1s
    df['devpF1'] = devpF1s
    df['devEntF1'] = devEntF1s
    df['devRelF1'] = devRelF1s
    df['testF1'] = testF1s
    df['testnF1'] = testnF1s
    df['testpF1'] = testpF1s
    df['testEntF1'] = testEntF1s
    df['testRelF1'] = testRelF1s
    df.to_csv("results/"+args.dataset+"_"+str(args.na_prob)+"_"+suffix+"_"+str(args.N)+"_"+str(args.K)+"_"+str(args.seed)+".csv")

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

if __name__ == "__main__":
    if not os.path.exists('checkpoints'):
        os.mkdir('checkpoints')

    if not os.path.exists('results'):
        os.mkdir('results')

    argv = sys.argv[1:]

    parser = Parser().getParser()
    args, _ = parser.parse_known_args(argv)

    set_seed(args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    tokenizer = BertTokenizer.from_pretrained(args.plm_path)
    bert = BertModel.from_pretrained(args.plm_path)

    print("Load {} data start...".format(args.dataset))
    dm = DataManager(tokenizer, args.N, args.K, args.dataset, args.na_prob)

    train_data, test_data, dev_data = dm.data['train'], dm.data['test'], dm.data['dev']
    print("train_data count: ", args.train_iter)
    print("test_data  count: ", args.test_iter)
    print("dev_data   count: ", args.dev_iter)

    model = Model(bert, args.N, args.K, tokenizer, True if args.na_prob != 0 else False, True if args.dataset == 'webnlg' else False)
    model.cuda()
    best_F1 = 0
    if args.start != '':
        pretrain_model = torch.load(args.start)
        if 'best_F1' in pretrain_model:
            best_F1 = pretrain_model['best_F1']
            model_dict = pretrain_model['net']
            print("best_F1:",best_F1)
        else:
            best_F1 = 0
            model_dict = model.state_dict()
        pretrained_dict = model_dict
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict) 
        model.load_state_dict(model_dict)
    model.share_memory()
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass
    # for name, param in model.named_parameters():
    #     print (name, param.size(), param.get_device())
    
    processes = []
    dataQueue = mp.Queue()
    resultQueue = mp.Queue()
    freeProcess = mp.Manager().Value("freeProcess", 0)
    lock = mp.Lock()
    flock = mp.Lock()
    print("Starting training service, overall process number: ", args.numprocess)
    for r in range(args.numprocess):
        p = mp.Process(target=worker, args= \
                (model, r, dataQueue, resultQueue, freeProcess, lock, flock, args.lr, args.alpha, args.beta, args.gamma))
        p.start()
        processes.append(p)
                
    if args.test == True:
        batchcnt = args.test_iter
        acc, cnt, tot = 0, 0, 0
        new_acc, new_cnt, new_tot = 0, 0, 0
        post_acc, post_tot, post_cnt = 0, 0, 0
        rel_acc, rel_cnt, rel_tot = 0, 0, 0
        ent_acc, ent_cnt, ent_tot = 0, 0, 0
        test_error_details = []
        test_error_statictis = {}
        model.eval()
        with torch.no_grad():
            for b in range(batchcnt):
                data = []
                for i in range(b * args.batchsize_test, (b+1) * args.batchsize_test):
                    data.append(test_data[i])
                acc_, cnt_, tot_, rel_acc_, rel_cnt_, rel_tot_, ent_acc_, ent_cnt_, ent_tot_, post_acc_, post_tot_, post_cnt_, new_acc_, new_cnt_, new_tot_, statictis, details = test(b, model, data, ["RE","NER"], \
                        dataQueue, resultQueue, freeProcess, lock, args.numprocess)
                acc += acc_
                cnt += cnt_ 
                tot += tot_
                new_acc += new_acc_
                new_cnt += new_cnt_ 
                new_tot += new_tot_
                post_acc += post_acc_
                post_cnt += post_cnt_
                post_tot += post_tot_
                rel_acc += rel_acc_
                rel_cnt += rel_cnt_ 
                rel_tot += rel_tot_
                ent_acc += ent_acc_
                ent_cnt += ent_cnt_ 
                ent_tot += ent_tot_
                update_statictis(statictis, test_error_statictis)
                test_error_details.extend(details)
            testF1 = calcF1(acc, cnt, tot)
            testnF1 = calcF1(new_acc, new_cnt, new_tot)
            testpF1 = calcF1(post_acc, post_cnt, post_tot)
            testRelF1 = calcF1(rel_acc, rel_cnt, rel_tot)
            testEntF1 = calcF1(ent_acc, ent_cnt, ent_tot)
            keys = list(test_error_statictis.keys())
            keys.sort()
            test_error_statictis = {k:test_error_statictis[k] for k in keys}
            print("test F1     : ", testF1)
            print("new test F1 : ", testnF1)
            print("post test F1: ", testpF1)
            print("test Rel F1 : ", testRelF1)
            print("test Rel F1 : ", testEntF1)
            print("test error  : ", test_error_statictis)
    else:
        work(["RE", "NER"], train_data, test_data, dev_data, model, args, args.epoch, best_F1)
    for p in processes:
        p.terminate()

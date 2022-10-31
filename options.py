import argparse

class Parser(object):
    def __init__(self):
        super(Parser, self).__init__()

    def getParser(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--lr', type=float, default=0.00002, help="Learning rate")
        parser.add_argument('--epoch', type=int, default=20, help="Number of epoch")
        parser.add_argument('--batchsize', type=int, default=4, help="Batch size on training")
        parser.add_argument('--batchsize_test', type=int, default=4, help="Batch size on testing")
        parser.add_argument('--print_per_batch', type=int, default=200, help="Print results every XXX batches")
        parser.add_argument('--numprocess', type=int, default=1, help="Number of process")
        parser.add_argument('--start', type=str, default='', help="Directory to load model")
        parser.add_argument('--test', action='store_true', help="Set to True to inference")
        parser.add_argument('--debug', action='store_true', default=False, help="Set to True to debug")
        parser.add_argument('--dataset', type=str, default='fewrel', help="Data directory")
        parser.add_argument('--gpu', type=str, default='1', help="gpuid")
        parser.add_argument('--N', type=int, default=5, help="N way")
        parser.add_argument('--K', type=int, default=1, help="K shot")
        parser.add_argument('--seed', type=int, default=5246, help="seed")
        parser.add_argument('--na_prob', type=float, default=0, help="na prob")
        parser.add_argument('--plm_path', type=str, default='/home/feijunbo/bert-base-uncased/', help="pretrain language model path")
        parser.add_argument('--train_iter', type=int, default=1000, help="train iter")
        parser.add_argument('--dev_iter', type=int, default=500, help="dev iter")
        parser.add_argument('--test_iter', type=int, default=500, help="test iter")
        parser.add_argument('--alpha', type=float, default=0.3, help="alpha")
        parser.add_argument('--beta', type=float, default=0.3, help="beta")
        parser.add_argument('--gamma', type=float, default=0.4, help="gamma")
        return parser
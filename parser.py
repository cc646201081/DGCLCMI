import argparse

def parse_args():

    parser = argparse.ArgumentParser(description="")

    parser.add_argument('--dataset', nargs='?', default='CMI-9589',
                        help='Choose a dataset from {CMI-9589, CMI-9905, CMI-20208}')

    parser.add_argument('--epoch', type=int, default=1000,
                        help='Number of epoch.')

    parser.add_argument('--embed_size', type=int, default=128, # 64
                        help='Embedding size.')

    parser.add_argument('--layer_size', nargs='?', default='[64,64,64]',
                        help='Output sizes of every layer')

    parser.add_argument('--batch_size', type=int, default=128, # 1024
                        help='Batch size.')

    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning rate.')

    parser.add_argument('--gpu_id', type=int, default=0)

    parser.add_argument('--node_dropout', nargs='?', default='[0.1]',
                        help='Keep probability w.r.t. node dropout (i.e., 1-dropout_ratio) for each deep layer. 1: no dropout.')

    parser.add_argument('--mess_dropout', nargs='?', default='[0.1,0.1,0.1]',
                        help='Keep probability w.r.t. message dropout (i.e., 1-dropout_ratio) for each deep layer. 1: no dropout.')

    return parser.parse_args()

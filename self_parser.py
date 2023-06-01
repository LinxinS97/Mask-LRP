import argparse


parser = argparse.ArgumentParser(description='Unsupervised distillation')
parser.add_argument('--num-process', default=10, type=int, help='number of process, set as 1 for debug')
parser.add_argument('--devices', default='1,2', type=str, help='list of GPU id, split by ","')
parser.add_argument('--dataset', default='qqp', type=str, help='name of the dataset')
parser.add_argument('--expl-method', default='generate_LRP', type=str, help='name of the explain method')
parser.add_argument('--synt-thres', default="0.49, 0.47, 0.82, 0.63", type=str, help='threshold for syntactic statistics')
parser.add_argument('--pos-thres', default=0.8, type=float, help='threshold for positional statistics')
parser.add_argument('--upos-thres', default=0.5, type=float, help='threshold for upos statistics')
parser.add_argument('--parsed-path', default='/home/linxin/xai/Transformer-Explainability/parse_res/', type=str, help='path to the parsed dataset')
parser.add_argument('--data-path', default='/home/linxin/data/', type=str, help='path to the original dataset')
parser.add_argument('--mask-type', default='synt_pos', choices=['synt', 'synt_pos', 'synt_upos', 'upos', 'upos_pos', 'pos', 'all', 'random', 'orig'], help='mask type use for LRP')
parser.add_argument('--save-path', default='./res', type=str, help='path for saving results')

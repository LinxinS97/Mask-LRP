import argparse


parser = argparse.ArgumentParser(description='Unsupervised distillation')
parser.add_argument('--num-process', default=10, type=int, help='number of process, set as 1 for debug')
parser.add_argument('--devices', default='1,2', type=str, help='list of GPU id, split by ","')
parser.add_argument('--dataset', default='qqp', type=str, help='name of the dataset')
parser.add_argument('--model-name', default='', type=str, help='name of the model')
parser.add_argument('--expl-method', default='generate_LRP', type=str, help='name of the explain method')
parser.add_argument('--synt-thres', default="0,0,0,0", type=str, help='threshold for syntactic statistics')
parser.add_argument('--pos-thres', default=0.8, type=float, help='threshold for positional statistics')
parser.add_argument('--parsed-path', default='/home/xai/Mask_LRP/parse_res/', type=str, help='path to the parsed dataset')
parser.add_argument('--data-path', default='/home/data/', type=str, help='path to the original dataset')
parser.add_argument('--mask-type', default='orig', choices=['synt', 'synt_pos', 'synt_pos_corruption', 'pos', 'all', 'random', 'random_abla', 'orig'], help='mask type use for LRP')
parser.add_argument('--save-path', default='./res', type=str, help='path for saving results')
parser.add_argument('--corruption-rate', default=0.0, type=float, help='(parameter for ablation study) control the corruption rate of head mask with randomly generated mask')
parser.add_argument('--repeat', default=1, type=int, help='set repeat times for random mask')

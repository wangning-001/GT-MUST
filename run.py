import argparse
import os
from model import GTMUST
from dataset import Dataset
from dataset_mpv import Dataset_MPV
from torch.utils.data import DataLoader

def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='../Datasets/viton_resize')
    parser.add_argument('--data_mpv', action='store_true')
    parser.add_argument('--data_mode', type=str, default='train')
    parser.add_argument('--stage', type=str, default='ILM', choices=['ILM', 'MWM', 'GTM'])
    parser.add_argument('--model_save_path', type=str, default='checkpoint_save_dir')
    parser.add_argument('--result_save_path', type=str, default='results_dir')
    parser.add_argument('--target_size', type=int, default=256)
    parser.add_argument('--num_iters', type=int, default=450000)
    parser.add_argument('--model_path', type=str, default="checkpoint_dir")
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--n_threads', type=int, default=6)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--gpu_id', type=str, default="0")
    parser.add_argument('--load_iter', type=int, default="200000")
    args = parser.parse_args()
        
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    model = GTMUST()
    if args.test:
        model.initialize_model(args.model_path, False, args.load_iter)
        model.cuda()
        if args.data_mpv:
            dataloader = DataLoader(Dataset_MPV(args.data_root, args.data_mode))
        else:
            dataloader = DataLoader(Dataset(args.data_root, args.data_mode))
        model.test(dataloader, args.result_save_path)
    else:
        model.initialize_model(args.model_path, True, args.load_iter)
        model.cuda()
        if args.data_mpv:
            dataloader = DataLoader(Dataset_MPV(args.data_root, args.data_mode), batch_size = args.batch_size, shuffle = True, num_workers = args.n_threads)
        else:
            dataloader = DataLoader(Dataset(args.data_root, args.data_mode), batch_size = args.batch_size, shuffle = True, num_workers = args.n_threads)
        model.train(dataloader, args.model_save_path, args.num_iters, args.stage)
if __name__ == '__main__':
    run()

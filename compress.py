import os
from pathlib import Path
import torch
import random

from dataset.datasets import build_dataset
from utils.parser import get_compress_args
from utils.utils import load_model

from SNP_compression.compress import snp

def get_imgs_to_calculate_snp(args):
    dataset_train, args.nb_classes = build_dataset(is_train=True, args=args)
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.num_imgs_snp_calculation,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    subset = torch.utils.data.Subset(dataset_train, [random.randint(0, len(dataset_train))-1 for _ in range(args.num_imgs_snp_calculation)])
    data_loader_calc_snp = torch.utils.data.DataLoader(subset, args.num_imgs_snp_calculation, num_workers=0, shuffle=False)
    inputs_to_calc = [list(data_loader_calc_snp)[0][0]]
    return inputs_to_calc

def compress_model_using_snp(args, model):
    inputs_to_calc = get_imgs_to_calculate_snp(args)
    model=snp(args, model, inputs_to_calc)
    save_path = os.path.join(args.output_dir, "compressed_model.pt")
    torch.save(model, save_path)
    return

if __name__=='__main__':
    args = get_compress_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    model = load_model(args)
    compress_model_using_snp(args, model)
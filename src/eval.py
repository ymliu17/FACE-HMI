import numpy as np
import argparse
from utils import set_seed, get_npy_files, calculate_metrics

argparser = argparse.ArgumentParser()
argparser.add_argument('--data_dir', type=str, default='data')
argparser.add_argument('--log_dir', type=str, default='results')
argparser.add_argument('--mod', type=str, default='ecg')
argparser.add_argument('--target', type=str, default='fatigue')

args = argparser.parse_args()

set_seed(42)

npy_files = get_npy_files(args.log_dir, args.mod, args.target)
npy_files = get_npy_files(
    log_dir=args.log_dir,
    mod=args.mod,
    target=args.target,
    rt_type=args.rt_type,
    interval=args.interval
)
print(f"Found {len(npy_files)} npy files in {args.log_dir}")

targets = []
outputs = []
for npy_file in npy_files:
    data = np.load(npy_file, allow_pickle=True).item()
    target = data['target']
    output = data['output']
    targets.append(target)
    outputs.append(output)

targets = np.concatenate(targets)
outputs = np.concatenate(outputs)

metrics_results = calculate_metrics(outputs, targets)
print(metrics_results)
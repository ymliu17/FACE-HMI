import torch
import argparse
import os
import numpy as np
from loader.ft_make import make_FACE_loader
from utils.utils import move_batch_to_device
import time

argparser = argparse.ArgumentParser()
argparser.add_argument('--data_dir', type=str, default='/Users/yangliu/Programs/FACE/FACE')
argparser.add_argument('--batch_size', type=int, default=1)
argparser.add_argument('--output_dir', type=str, default='/Users/yangliu/Programs/FACE/results_face')

args = argparser.parse_args()

# Load the pretrained model

model = torch.load('/Users/yangliu/Programs/FACE/results_face/target-focus/sub-F4001/RVT/model.pth', weights_only=False, map_location=torch.device('cpu'))['model']
model.eval()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(" device = ", device)
model.to(device)

# Create output directory if it doesn't exist
os.makedirs(args.output_dir, exist_ok=True)

for subject in ['F4001', 'F4002', 'F4003', 'F4004', 'F4005', 'F4006', 'F4007', 'F4008', 'F4009', 'F4010', 'F4011', 'F4012', 'F4013', 'F4014', 'F4015', 'F4016', 'F4017', 'F4018', 'F4019', 'F4020']:

    start_time = time.time()

    # Load the data
    _, test_loader = make_FACE_loader(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        test_subject=subject
    )

    # Check if the test_loader is empty
    if len(test_loader.dataset) == 0:
        raise ValueError(f"No data found for test subject {subject}")

    print(f"Number of samples in test dataset: {len(test_loader.dataset)}")

    # Iterate over test_loader and save results block by block
    results = []
    with torch.no_grad():
        for batch in test_loader:
            batch = move_batch_to_device(batch, device)
            blocks = batch['idx']
            faces = batch['seq']
            
            outputs = []
            h_t = torch.zeros(2, 1, 1024).to(device)
            for i in range(faces.size(1)):
                block_faces = faces[:, i, :, :, :]
                output, h_t = model(block_faces, h_t)
                outputs.append(output.cpu().numpy())
            
            outputs = np.concatenate(outputs, axis=0)
            for block_idx, output in zip(blocks, outputs):
                results.append({'Sub_Ses_Block': block_idx, 'focus': torch.sigmoid(torch.tensor(output[1])).item()})

    # Save results to a file
    output_file = os.path.join(args.output_dir, 'target-focus', f'focus_{subject}.npy')
    np.save(output_file, results)
    print(f"Results saved to {output_file}")
    data = np.load(output_file, allow_pickle=True)
    print(data)

    end_time = time.time()
    print(f"Time taken for subject {subject}: {end_time - start_time:.2f} seconds")
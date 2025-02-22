import os
import time

import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt

from splitquant import SplitQuant

# torch.onnx.export(model, dummy_input, f"{model_name}_original.onnx",
#                   training=TrainingMode.EVAL,
#                   do_constant_folding=True)  # use `False` to see BatchNorm

def plot_k_means_clustering(k: int, original_kernels: nn.parameter.Parameter, first_n: int):
    assert k > 0, "K should be a positive integer."
    assert original_kernels is not None, "Kernels cannot be None."

    kernels_clone = original_kernels.data.clone()
    for i in range(len(kernels_clone)):
        if first_n < i + 1:
            break

        # k-means clustering
        kernel = kernels_clone[i]
        kernel_flat = kernel.view(-1).cpu().numpy().reshape(-1, 1)  # Reshape for K-means
        centers, boundaries, labels = SplitQuant.get_k_result(k, kernel_flat)

        # plot
        plt.figure(figsize=(10, 6))
        plt.scatter(centers, np.zeros_like(centers), c='red', marker='X', s=200, label='Cluster Centers')
        plt.scatter(kernel_flat, np.zeros_like(kernel_flat), c=labels, cmap='viridis', marker='o', label='Weights')
        for midpoint in boundaries:
            plt.axvline(midpoint, color='gray', linestyle='--', linewidth=1)

        plt.title('K-means Clustering of Weights')
        plt.xlabel('Weight Values')
        plt.yticks([])  # Hide y-axis ticks
        plt.legend()
        plt.show()


def evaluate_custom(model, data, target, idx2id, device):
    model.to(device).eval()
    total_time, correct = 0, 0
    with torch.no_grad():
        for img, target in zip(data, target):
            start = time.time()
            output = model(img.to(device))
            end = time.time()
            delta = end - start
            total_time += delta
            pred_idx = int(output[0].sort()[1][-1:])
            pred_id = idx2id[pred_idx]
            if target == pred_id:
                correct += 1
    inference_time = total_time/len(data)
    accuracy = (correct/len(data))*100
    return inference_time, accuracy


def print_model_size(mdl):
    torch.save(mdl.state_dict(), "tmp.pt")
    print("%.2f MB" %(os.path.getsize("tmp.pt")/1e6))
    os.remove('tmp.pt')


def compare_model_outputs(model_1: torch.nn.Module,
                          model_2: torch.nn.Module,
                          input_data: torch.Tensor,
                          tolerance: float = 1e-5) -> tuple[bool, torch.Tensor]:
    output_1 = model_1(input_data)
    output_2 = model_2(input_data)
    are_close = torch.allclose(output_1, output_2, atol=tolerance)
    abs_diff = torch.abs(output_1 - output_2)
    return are_close, abs_diff


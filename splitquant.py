import copy
import torch
import torch.nn as nn
import numpy as np
from torch.nn.quantized import FloatFunctional
from sklearn.cluster import KMeans


class SplitQuant:
    def __init__(self, device: str = "cpu", split_actv: bool = False):
        self._device = device
        self._split_actv = split_actv
        print(f"SplitQuant is running on {self._device}, ", end="")
        if self._split_actv:
            print("Split weights and activations")
        else:
            print("Split weights only")

    def split(self, model: nn.Module, k: int):
        layers = self.get_layer_list(model)
        for i, layer in enumerate(layers):
            if self._split_actv and self.is_activation(layer):
                split_layers = SplitLayersActivation(layer)
                self.replace_layer(model, layer, split_layers)
            elif self.has_weight_for_split(layer, k):
                split_layers = SplitLayersWeightBias(layer)

                # weight
                if isinstance(layer, nn.Conv2d) or "Conv2d" in type(layer).__name__:
                    weight_lower, weight_mid, weight_upper = self.k_means_split_kernels(k, layer.weight)
                elif isinstance(layer, nn.Linear) or "Linear" in type(layer).__name__:
                    weight_lower, weight_mid, weight_upper = self.k_means_split_1d(k, layer.weight, self._device)
                else:
                    print(f"WARNING: {layer}, {type(layer)} has weights but cannot be split.")
                with torch.no_grad():
                    split_layers.lower.weight.copy_(weight_lower)
                    split_layers.middle.weight.copy_(weight_mid)
                    split_layers.upper.weight.copy_(weight_upper)

                # bias
                if hasattr(layer, "bias"):
                    if layer.bias is not None:
                        if torch.all(layer.bias == 0):
                            # empty bias
                            print("Bias is empty so it is not split.")
                            continue
                        bias_lower, bias_mid, bias_upper = self.k_means_split_1d(k, layer.bias, self._device)
                        with torch.no_grad():
                            split_layers.lower.bias.copy_(bias_lower)
                            split_layers.middle.bias.copy_(bias_mid)
                            split_layers.upper.bias.copy_(bias_upper)

                # replace the layer
                self.replace_layer(model, layer, split_layers)

    def replace_layer(self, model: nn.Module, old_layer, new_layer):
        for name, module in model.named_children():
            if module == old_layer:
                setattr(model, name, new_layer)
            else:
                # Recursively check in child modules
                self.replace_layer(module, old_layer, new_layer)

    @staticmethod
    def get_layer_list(module: nn.Module):
        output = []
        children = list(module.children())
        if len(children) == 0:
            output.append(module)
        for child in children:
            output.extend(SplitQuant.get_layer_list(child))
        return output

    @staticmethod
    def is_activation(layer):
        activations = (nn.SiLU, nn.ReLU, nn.ReLU6, nn.Sigmoid, nn.Tanh, nn.LeakyReLU, nn.ELU, nn.Softmax)
        return isinstance(layer, activations)

    @staticmethod
    def has_weight_for_split(layer, k):
        if hasattr(layer, "weight"):
            # Normalizations have weights, but they are in fact Gamma
            if "Norm" in type(layer).__name__:
                # nn.modules.batchnorm._NormBase
                # nn.modules.normalization.LayerNorm
                # transformers.models.llama.modeling_llama.LlamaRMSNorm
                # etc.
                return False

            # Do not split embeddings
            if "Embedding" in type(layer).__name__:
                return False

            ignore = ( # put here layer types you want to ignore
                nn.modules.linear.NonDynamicallyQuantizableLinear,
            )
            if isinstance(layer, ignore):
                return False
            return len(layer.weight) >= k
        return False

    @staticmethod
    def k_means_split_kernels(k: int, original_kernels: nn.parameter.Parameter):
        assert k > 0, "K should be a positive integer."
        assert original_kernels is not None, "Kernels cannot be None."

        kernels_clone = original_kernels.data.clone()
        kernels_lower = []
        kernels_middle = []
        kernels_upper = []

        for kernel in kernels_clone:
            weight_lower, weight_mid, weight_upper = SplitQuant._k_means_split(k, kernel)
            kernels_lower.append(weight_lower)
            kernels_middle.append(weight_mid)
            kernels_upper.append(weight_upper)

        kernels_lower = torch.stack(kernels_lower)
        kernels_middle = torch.stack(kernels_middle)
        kernels_upper = torch.stack(kernels_upper)
        return kernels_lower, kernels_middle, kernels_upper

    @staticmethod
    def k_means_split_1d(k: int, weight_or_bias: nn.parameter.Parameter, device: str):
        assert k > 0, "K should be a positive integer."
        assert weight_or_bias is not None, "Weight or bias cannot be None."
        cloned_data = weight_or_bias.data.clone()
        return SplitQuant._k_means_split(k, cloned_data, device)

    @staticmethod
    def _k_means_split(k: int, cloned_data: torch.Tensor, device: str):
        # k-means clustering
        # contiguous() to collect scattered data that are separated on the memory
        data_flat = cloned_data.contiguous().view(-1).cpu().numpy().reshape(-1, 1)
        _, boundaries, _ = SplitQuant.get_k_result(k, data_flat)
        boundaries = boundaries.to(device)

        if len(boundaries) > len(set(boundaries)):
            print("WARNING: Your k value is too big for your data.")

        # split the bias
        data_lower = torch.where(cloned_data <= boundaries[0], cloned_data, torch.tensor(0.0))
        data_middle = torch.where(boundaries[0] < cloned_data, cloned_data, torch.tensor(0.0))
        data_middle = torch.where(data_middle < boundaries[1], data_middle, torch.tensor(0.0))
        data_upper = torch.where(boundaries[1] <= cloned_data, cloned_data, torch.tensor(0.0))

        return data_lower, data_middle, data_upper

    @staticmethod
    def get_k_result(k: int, data):
        kmeans = KMeans(n_clusters=k, n_init="auto")
        kmeans.fit(data)
        centers = np.sort(kmeans.cluster_centers_, axis=0)
        boundaries = torch.tensor((centers[:-1] + centers[1:]) / 2)
        return centers, boundaries, kmeans.labels_


class SplitLayersActivation(nn.Module):
    def __init__(self, original_layer, k=3):
        super().__init__()
        self.k = k
        self.activ_lower = None
        self.activ_middle = None
        self.activ_upper = None
        activations = [nn.ReLU, nn.Tanh, nn.GELU, nn.SiLU]
        for actv in activations:
            if isinstance(original_layer, actv):
                self.activ_lower = actv()
                self.activ_middle = actv()
                self.activ_upper = actv()
        self.f_concat = FloatFunctional()

    def forward(self, x):
        # k should be equal to no. of clusters (e.g., 3)
        # dim=1 means over the channels
        lower, middle, upper = torch.tensor_split(x, self.k, dim=1)
        act_lower = self.activ_lower(lower)
        act_middle = self.activ_middle(middle)
        act_upper = self.activ_upper(upper)
        concat = self.f_concat.cat([act_lower, act_middle, act_upper], dim=1)
        return concat


class SplitLayersWeightBias(nn.Module):
    def __init__(self, original_layer):
        super().__init__()
        assert original_layer.weight is not None, "Original layers should have weights."
        assert len(original_layer.weight) > 0, "Original layer has empty weight."

        self.lower = copy.deepcopy(original_layer)
        self.middle = copy.deepcopy(original_layer)
        self.upper = copy.deepcopy(original_layer)
        with torch.no_grad():
            for new_layer in [self.lower, self.middle, self.upper]:
                nn.init.zeros_(new_layer.weight)
                if hasattr(new_layer, "bias"):
                    if new_layer.bias is not None:
                        nn.init.zeros_(new_layer.bias)

        self.f_add_step1 = FloatFunctional()
        self.f_add_step2 = FloatFunctional()

    def forward(self, x):
        out1 = self.lower(x)
        out2 = self.middle(x)
        out3 = self.upper(x)
        add_12 = self.f_add_step1.add(out1, out2)
        add_123 = self.f_add_step2.add(add_12, out3)
        return add_123

import torch
from torch._C._onnx import TrainingMode
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models import vit_b_16, ViT_B_16_Weights
from splitquant import SplitQuant


def test_output_same() -> bool:
    model_names = ["resnet18", "vit_b_16"]
    for model_name in model_names:
        model = None
        if model_name == "resnet18":
            model = resnet18(weights=ResNet18_Weights.DEFAULT)
        elif model_name == "vit_b_16":
            model = vit_b_16(weights=ViT_B_16_Weights)
        assert model is not None

        # for k-means clustering
        k = 3

        # original model
        dummy_input = torch.randn(1, 3, 224, 224)
        output_original = model(dummy_input)
        print(model)
        torch.onnx.export(model, dummy_input, f"{model_name}_original.onnx",
                          training=TrainingMode.EVAL,
                          do_constant_folding=True)  # use `False` to see BatchNorm

        # SplitQuant model
        layer_splitter = SplitQuant()
        layer_splitter.split(model, k)
        output_replaced = model(dummy_input)
        print(model)
        torch.onnx.export(model, dummy_input, f"{model_name}_splitquant.onnx",
                          training=TrainingMode.EVAL,
                          do_constant_folding=True)

        # whether the outputs are same, or close
        tolerance = 1e-05
        are_close = torch.allclose(output_original, output_replaced, atol=tolerance)
        diffs = torch.abs(output_original - output_replaced)
        print(f"are_close: {are_close}, max: {diffs.max()}, min: {diffs.min()}, median: {diffs.median()})")

        return are_close

if __name__ == "__main__":
    print(test_output_same())
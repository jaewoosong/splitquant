import transformers
from transformers import pipeline
from transformers import AutoModelForSequenceClassification
from optimum.quanto import quantize, freeze
from optimum.quanto.tensor.qtype import qint
from datasets import load_dataset
from huggingface_hub import login
from splitquant import SplitQuant

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def load_bert_tiny_model(model_id, splitquant=False, quantization=False, qbits=2):
    model = AutoModelForSequenceClassification.from_pretrained(model_id)
    if splitquant:
        ls = SplitQuant()
        ls.split(model, k=3)
    if quantization:
        quantize(model, weights=qint(qbits), activations=None)
        freeze(model)
    return model

def run(pipeline, model, data_inputs, data_labels):
    pipeline.model = model
    num_total = len(data_inputs)
    num_correct = 0
    preds = pipeline(data_inputs)
    assert num_total == len(preds) == len(data_inputs) == len(data_labels)
    for i in range(len(preds)):
        curr_pred = preds[i]
        pred_label = curr_pred["label"]
        if isinstance(pred_label, str):
            assert pred_label[:-1] == "LABEL_", f"{pred_label} is malformed."
            if int(pred_label[-1]) == data_labels[i]:
                num_correct += 1
        else:
            if pred_label == data_labels[i]:
                num_correct += 1
    print(f"Accuracy: {num_correct/num_total} ({num_correct}/{num_total})")
    return num_correct, num_total

if __name__ == "__main__":
    hf_token = "your_huggingface_token_here"
    try:
        login(token=hf_token)
    except:
        print(f"Your HuggingFace token ({hf_token}) is invalid.")
        print("If the code does not run, please check your HuggingFace token.\n")

    print("Starting...\n")
    kSEED = 12345
    transformers.set_seed(kSEED)

    kDEVICE = "cpu"
    category = "text-classification"
    datasets = ["spam_detection", "emotion"]
    scenario = datasets[1]  # 0: spam_detection / 1: emotion

    model_id: str = ""
    dataset_id: str = ""
    the_dataset = None
    data_text = None
    data_label = None

    if scenario == "spam_detection":
        model_id = "mrm8488/bert-tiny-finetuned-sms-spam-detection"
        dataset_id = "ucirvine/sms_spam"
        the_dataset = load_dataset(dataset_id)
        data_text = the_dataset["train"]["sms"]
        data_label = the_dataset["train"]["label"]
    elif scenario == "emotion":
        model_id = "gokuls/BERT-tiny-emotion-intent"
        dataset_id = "dair-ai/emotion"
        the_dataset = load_dataset(dataset_id)
        data_text = the_dataset["test"]["text"]
        data_label = the_dataset["test"]["label"]
    else:
        assert scenario in datasets

    print(f"Model: {model_id}")
    print(f"Dataset: {dataset_id}")
    print(f"Device: {kDEVICE}")

    # Default
    classification_pipeline = pipeline(category, model=model_id, device=kDEVICE)
    print("\nCreating the pipeline...")
    classification_pipeline(data_text)

    # Default but load model manually
    print("\nBaseline:")
    run(classification_pipeline,
        load_bert_tiny_model(model_id),
        data_text,
        data_label)

    # SplitQuant Floating-Point
    print("\nSplitQuant Floating-Point:")
    run(classification_pipeline,
        load_bert_tiny_model(model_id, splitquant=True),
        data_text,
        data_label)

    for q in [2, 4, 8]:
        # Default, quantization
        print(f"\nQuantization ({q}-bit):")
        run(classification_pipeline,
            load_bert_tiny_model(model_id, quantization=True, qbits=q),
            data_text,
            data_label)

        # SplitQuant, quantization
        print(f"\nSplitQuant & Quantization ({q}-bit):")
        run(classification_pipeline,
            load_bert_tiny_model(model_id, splitquant=True, quantization=True, qbits=q),
            data_text,
            data_label)

    print("\nSplitQuant evaluation is finished.")

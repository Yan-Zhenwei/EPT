from datasets import load_dataset, Dataset, load_from_disk
from tqdm import tqdm
import os
import json

commonsense_root = os.path.join(os.path.dirname(__file__), "commonsense")

def save_dataset_splits(task_name, splits_data):
    out_dir = os.path.join(commonsense_root, f"{task_name}_with_prompt")
    os.makedirs(out_dir, exist_ok=True)
    splits = []
    for split_name, ds in splits_data.items():
        split_path = os.path.join(out_dir, split_name)
        ds.save_to_disk(split_path)
        splits.append(split_name)
    with open(os.path.join(out_dir, "dataset_dict.json"), "w") as f:
        json.dump({"splits": splits}, f)

def process_scitail():
    raw = load_dataset("scitail", "snli_format")
    def label_map(l):
        return {0: "entailment", 1: "neutral"}.get(int(l), "-1")
    tmpl = "scitail premise: {premise} hypothesis: {hypothesis}"
    splits = {}
    for split in ["train", "validation"]:
        data = raw[split]
        prompts = []
        labels = []
        for ex in tqdm(data):
            prompts.append(tmpl.format(premise=ex["premise"], hypothesis=ex["hypothesis"]))
            labels.append(label_map(ex["label"]))
        ds = Dataset.from_dict({"prompt": prompts, "label": labels})
        ds = ds.filter(lambda x: x["label"] != "-1")
        splits[split] = ds
    save_dataset_splits("scitail", splits)

def process_cb():
    raw = load_dataset("super_glue", "cb")
    def label_map(l):
        return {0: "entailment", 1: "contradiction", 2: "neutral"}.get(int(l), "-1")
    tmpl = "cb premise: {premise} hypothesis: {hypothesis}"
    splits = {}
    for split in ["train", "validation"]:
        data = raw[split]
        prompts = []
        labels = []
        for ex in tqdm(data):
            prompts.append(tmpl.format(premise=ex["premise"], hypothesis=ex["hypothesis"]))
            labels.append(label_map(ex["label"]))
        ds = Dataset.from_dict({"prompt": prompts, "label": labels})
        ds = ds.filter(lambda x: x["label"] != "-1")
        splits[split] = ds
    save_dataset_splits("cb", splits)

if __name__ == "__main__":
    process_scitail()
    process_cb()

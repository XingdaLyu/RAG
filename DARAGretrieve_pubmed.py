import os
import json
import random
import argparse
import pandas as pd
from tqdm import tqdm

from retrieve.retriever import bm25_retrieve
from root_dir_path import ROOT_DIR

from datasets import load_dataset

random.seed(42)

def main(dataset, dataset_name, topk=3):
    output_dir = os.path.join(ROOT_DIR, "data_retrieval", dataset_name)
    os.makedirs(output_dir, exist_ok=True)

    print("### Loading dataset ###")
    solve_dataset = dataset
    
    ret = []
    for data in tqdm(solve_dataset):
        passages = bm25_retrieve(data["QUESTION"], topk=topk+10)
        data["passages"] = passages
        ret.append(data)
    with open(os.path.join(output_dir, "retrieval.json"), "w") as fout:
        json.dump(ret, fout, indent=4)


if __name__ == "__main__":
    dataset = load_dataset('bigbio/pubmed_qa')
    main(dataset["validation"], 'pubmed_qa_val')

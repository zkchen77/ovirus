# -*- coding: utf-8 -*-
"""
07_pubtator3_genes.py
@author: zkchen

=====================
1. Literature-derived disease-associated gene sets via the PubTator3 API.
--------------------
For each skeletal disorder the script queries

    GET https://www.ncbi.nlm.nih.gov/research/pubtator3-api/relations?
    e1=@DISEASE_<MeSH>&e2=gene&format=json
    
    MeSH = [
        "Low_Back_Pain", "Osteoarthritis", 
        "Osteonecrosis", "Osteoporosis", 
        "Scoliosis", "Spinal_Stenosis", 
        "Spondylolisthesis", "Spondylosis"
        ]

2. Co-occurrence (intersection) gene sets for the three disease patterns.
--------------------
    patternI   = Low back pain  & Osteoarthritis    & Osteonecrosis
    patternII  = Spinal stenosis & Osteoporosis     & Spondylolisthesis
    patternIII = Scoliosis       & Spondylosis
    
=====================

"""

import requests
import json
import csv
import os
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "pubtator3")
os.makedirs(OUT, exist_ok=True)
OUT2 = os.path.join(ROOT, "data", "string")
os.makedirs(OUT2, exist_ok=True)

def from_pubmed_get_relation(query, para):
    base = "https://www.ncbi.nlm.nih.gov/research/pubtator3-api/relations?"
    url = "{url}e1=@{e1}&e2={e2}&format=json".format(url=base, e1=query, e2=para)
    
    r = requests.get(url)
    data = json.loads(r.text)
    
    print(url)
    print(r, len(data))
    
    csv_path = os.path.join(OUT, "{}+{}.csv".format(query, para))
    with open(csv_path, mode='w', newline='', encoding='utf-8') as csvfile:
        fieldnames = data[0].keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in data:
            writer.writerow(row)
    print(query, "Done!")
    return None

def load_genes(disease):
    csv_path = os.path.join(OUT, f"{disease}+gene.csv")
    df = pd.read_csv(csv_path, usecols=["target"])
    symbols = (df["target"].dropna().str.removeprefix("@GENE_").str.replace("_", "-", regex=False))
    return set(symbols)


def write_lines(path, items, newline="\n"):
    """Write an iterable to a UTF-8 text file, one item per line."""
    with open(path, "w", encoding="utf-8", newline=newline) as fh:
        fh.write(newline.join(items) + newline)


def main():
    DiseaseList = [
        "DISEASE_Low_Back_Pain", "DISEASE_Osteoarthritis", 
        "DISEASE_Osteonecrosis", "DISEASE_Osteoporosis", 
        "DISEASE_Scoliosis", "DISEASE_Spinal_Stenosis", 
        "DISEASE_Spondylolisthesis", "DISEASE_Spondylosis"
        ]

    PATTERNS = {
        "I":   ["DISEASE_Low_Back_Pain", "DISEASE_Osteoarthritis",
                "DISEASE_Osteonecrosis"],
        "II":  ["DISEASE_Spinal_Stenosis", "DISEASE_Osteoporosis",
                "DISEASE_Spondylolisthesis"],
        "III": ["DISEASE_Scoliosis", "DISEASE_Spondylosis"],
    }
    
    for disease in DiseaseList:
        from_pubmed_get_relation(disease, "gene")

    genes_per_disease = {d: load_genes(d) for d in DiseaseList}

    for name, diseases in PATTERNS.items():
        genes = sorted(set.intersection(*(genes_per_disease[d]
                                          for d in diseases)))
        path = os.path.join(OUT2, f"pattern{name}_co-occurrence genes.txt")
        write_lines(path, genes)
        print(f"pattern{name}: {len(genes):>3} genes -> {os.path.basename(path)}")

if __name__ == "__main__":
    main()



#%%
# Goal: Identify neighbor genes of each target, check the overlap of all targets. check the same for dominant genes.
# Run the same analysis for MDM2 and the TP53 pathway players. Is there uniqueness for any of them or is the overlap close to 100%?

import argparse
import traceback
import warnings
import pandas as pd
import numpy as np

from numpy import nan
import Parse_data as Parse
import itertools
import csv
import sklearn
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestClassifier
# from models.lr import Logistic_Regression
from models.mlp import MLP
from models.gcn import GCN
from data.datasets import TCGADataset
from data.gene_graphs import GeneManiaGraph, RegNetGraph, HumanNetV2Graph, \
    FunCoupGraph

from data.utils import record_result
from tqdm import tqdm
import pickle
import networkx as nx


# Path to save results
PATH_save_results = '/home/shair/Desktop/Gil/2024_STAMP_Explainatory_Translational_Sorafenib/Data_generated/Sorafenib_first_degree_neighbors_df/'
# Read in data: TCGA
dataset = TCGADataset()
# Remove duplicate genes!
dataset.df = dataset.df.loc[:,~dataset.df.columns.duplicated()]
# save a copy of dataset, since it will be modified for cancer_type specific samples
dataset_df_copy = dataset.df

# graphs dictionary
graph_dict = {"regnet": RegNetGraph, "genemania": GeneManiaGraph,
                  "humannetv2": HumanNetV2Graph, "funcoup": FunCoupGraph}
# load graph once for funcoup
gene_graph_funcoup = graph_dict["funcoup"]()
# humannetv2
gene_graph_humannetv2 = graph_dict["humannetv2"]()

# Important genes list.
Sorafenib_targets = ["BRAF", "RAF1", "FGFR1", "KIT", "FLT3", "RET", "PDGFRB"] # PDGFR = "PDGFRB", VEGFR1 = "FLT1", VEGFR2 = "KDR", VEGFR3 = "FLT4"
RTK_RAS_dominant_genes = ["EGFR", "ERRFI1", "KRAS", "MET", "NF1", "RASA1"]
# TP53_pathway_genes = ["TP53", "MDM2", "MDM4", "CDKN2A", "ATM", "CHEK2", "RPS6KA3"]


# # Pathway dictionary for labeling
# pathway_dict = {"Cell Cycle": "CDKN2A", "HIPPO": "LATS1", "MYC": "MYC", "NOTCH": "NOTCH1", "NRF2": "CUL3", "PI3K": "PTEN", "RTK RAS": "KRAS", "TP53": "TP53",
#                 "TGF-Beta":	"TGFBR1", "WNT": "APC"}
# gene_tumor_type_targets_Sorafenib_project = {
#     # "BRAF": ["pan_cancer", "LUAD"],
#     # "EGFR": ["pan_cancer", "LIHC", "BRCA"],
#     # "KRAS": ["pan_cancer", "LIHC", "BRCA"],
#     # "MET": ["pan_cancer", "LIHC", "BRCA"],
#     # "NF1": ["pan_cancer", "LIHC", "BRCA"],
#     # "RASA1": ["pan_cancer", "LIHC", "BRCA"],
#     # "ERRFI1": ["pan_cancer", "LIHC", "BRCA"],
#     # "FGFR1": ["pan_cancer", "BRCA", "LUSC", "HNSC"]
#     "RAF1": ["pan_cancer"],
#     "PDGFR": ["pan_cancer"],
#     "KIT": ["pan_cancer"],
#     "FLT3": ["pan_cancer"],
#     "RET": ["pan_cancer"]
# }
# Old code lines:
# seed = 1234
# cuda = torch.cuda.is_available()
# # tuning
# data = "pathway_cell_paper"
# pathway_type = "GENE"  # can also be "PATHWAY" or "ALTERATION"
# dropout = False
# agg = "hierarchy"
# embedding = 30
# # Load relevant
# load all study IDs in TCGA (n = 33)
# Study_IDs = Parse.get_cbioportal_TCGAbiolinks_cancer_types()
# Study_IDs.append("pan_cancer")
# pathway_list = gene_tumor_type_targets_Sorafenib_project.keys()
# Dictionary for combinations to train on
# load pathway list
# pathway_list = Parse.get_cell_paper_pathway_list(pathway_type)
# pathway_list = [x for x in pathway_list if x != "PI3K"]
# pathway_list.insert(0, "RTK RAS")


# Example neighbor generation functions
def funcoup_neighbors(gene):
    neighbors_funcoup = list(gene_graph_funcoup.first_degree(gene)[0])
    neighbors_funcoup = [n for n in neighbors_funcoup if n in dataset_df_copy.columns.values]

    # modify adj for neighbors- funcoup
    # get subgraph for distinct genes
    neighborhood_funcoup = gene_graph_funcoup.nx_graph.subgraph(neighbors_funcoup).copy()
    # get adj for distinct nodes
    adj_funcoup = np.asarray(nx.to_numpy_matrix(neighborhood_funcoup))
    # SUBGRAPH NODES ARE NOT ORDERED- ORDER THEM using this line of code.
    neighbors_funcoup = list(neighborhood_funcoup.nodes)
    return neighbors_funcoup

def humannetv2_neighbors(gene):
    # adj_humannetv2 = gene_graph_humannetv2.adj()
    neighbors_humannetv2 = list(gene_graph_humannetv2.first_degree(gene)[0])
    neighbors_humannetv2 = [n for n in neighbors_humannetv2 if n in dataset_df_copy.columns.values]

    # modify adj for neighbors- humannet
    # get subgraph for distinct genes
    neighborhood_humannetv2 = gene_graph_humannetv2.nx_graph.subgraph(neighbors_humannetv2).copy()
    # get adj for distinct nodes
    adj_humannetv2 = np.asarray(nx.to_numpy_matrix(neighborhood_humannetv2))
    # SUBGRAPH NODES ARE NOT ORDERED- ORDER THEM using this line of code.
    neighbors_humannetv2 = list(neighborhood_humannetv2.nodes)

    return neighbors_humannetv2

# Function to generate a dataframe for a list of genes
def generate_dataframe(genes, neighbor_function):
    data = {gene: neighbor_function(gene) for gene in genes}
    return pd.DataFrame.from_dict(data, orient="index").transpose()

def process_dataframe_with_named_rows(df):
    """
    Adds:
    1. First row: Number of total neighbors in each column.
    2. Second row: Number of unique neighbors in each column.
    3. New column at the end: "Core Neighbors" (neighbors common to all columns).
    """
    # Compute total neighbors and unique neighbors for each column
    total_counts = {}
    unique_counts = {}
    all_neighbors = [set(df[col].dropna()) for col in df.columns]  # Drop NaN for empty rows

    for i, col in enumerate(df.columns):
        total_counts[col] = len(all_neighbors[i])  # Total neighbors in the column
        others = set.union(*(all_neighbors[:i] + all_neighbors[i + 1:]))  # Neighbors in other columns
        unique_counts[col] = len(all_neighbors[i] - others)  # Unique to this column

    # Add the total and unique neighbor counts as the first two rows
    total_counts_row = [total_counts[col] for col in df.columns]
    unique_counts_row = [unique_counts[col] for col in df.columns]

    # Insert rows
    df.loc['Total Neighbors'] = total_counts_row
    df.loc['Unique Neighbors'] = unique_counts_row
    df = df.reset_index().rename(columns={'index': 'Row Names'})  # Reset index to manage row names

    # Compute core neighbors (intersection of all columns)
    core_neighbors = set.intersection(*all_neighbors)
    core_neighbors_column = [", ".join(core_neighbors) if i < len(core_neighbors) else None
                             for i in range(len(df))]
    df["Core Neighbors"] = pd.Series(core_neighbors_column)

    return df

# Combine Sorafenib targets with RTK-RAS dominant genes
sorafenib_rtk_ras_dominant = Sorafenib_targets + RTK_RAS_dominant_genes

# Create dataframes for FunCoup
funcoup_sorafenib_df = generate_dataframe(Sorafenib_targets, funcoup_neighbors)
funcoup_rtk_ras_df = generate_dataframe(RTK_RAS_dominant_genes, funcoup_neighbors)
funcoup_sorafenib_rtk_ras_dominant_df = generate_dataframe(sorafenib_rtk_ras_dominant, funcoup_neighbors)

# Create dataframes for HumanNetV2
humannetv2_sorafenib_df = generate_dataframe(Sorafenib_targets, humannetv2_neighbors)
humannetv2_rtk_ras_df = generate_dataframe(RTK_RAS_dominant_genes, humannetv2_neighbors)
humannetv2_sorafenib_rtk_ras_dominant_df = generate_dataframe(sorafenib_rtk_ras_dominant, humannetv2_neighbors)





# Process each dataframe
funcoup_sorafenib_df = process_dataframe_with_named_rows(funcoup_sorafenib_df)
funcoup_rtk_ras_df = process_dataframe_with_named_rows(funcoup_rtk_ras_df)
funcoup_sorafenib_rtk_ras_dominant_df = process_dataframe_with_named_rows(funcoup_sorafenib_rtk_ras_dominant_df)

humannetv2_sorafenib_df = process_dataframe_with_named_rows(humannetv2_sorafenib_df)
humannetv2_rtk_ras_df = process_dataframe_with_named_rows(humannetv2_rtk_ras_df)
humannetv2_sorafenib_rtk_ras_dominant_df = process_dataframe_with_named_rows(humannetv2_sorafenib_rtk_ras_dominant_df)

# Optionally, save the dataframes to CSV files
funcoup_sorafenib_df.to_csv(PATH_save_results+"funcoup_sorafenib.csv", index=False)
funcoup_rtk_ras_df.to_csv(PATH_save_results+"funcoup_rtk_ras.csv", index=False)
funcoup_sorafenib_rtk_ras_dominant_df.to_csv(PATH_save_results+"funcoup_Sorafenib_rtk_ras.csv", index=False)

humannetv2_sorafenib_df.to_csv(PATH_save_results+"humannetv2_sorafenib.csv", index=False)
humannetv2_rtk_ras_df.to_csv(PATH_save_results+"humannetv2_rtk_ras.csv", index=False)
humannetv2_sorafenib_rtk_ras_dominant_df.to_csv(PATH_save_results+"humannetv2_Sorafenib_rtk_ras.csv", index=False)





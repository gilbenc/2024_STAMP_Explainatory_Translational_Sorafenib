import pandas as pd
import torch
import os, fnmatch
import numpy as np
from data.datasets import TCGADataset
from matplotlib import pyplot as plt
import scipy
import math
import pickle
from data.gene_graphs import GeneManiaGraph, RegNetGraph, HumanNetV2Graph, \
    FunCoupGraph
import networkx as nx
from data.datasets import TCGADataset


### ToDo: modify parsing functions for GDSC, CCLE ###

PATH = "/home/shair/Desktop/Gil/2024_STAMP_Explainatory_Translational_Sorafenib/"
PATH_output = "/home/shair/Desktop/Gil/2024_STAMP_Explainatory_Translational_Sorafenib/Data_generated/Perturbation_models/"
results_PATH = PATH_output + "STAMP_predictions/"




### Functions

# returns model of model_type, for cancer_type & gene combination
def load_model(model_type, cancer_type, gene, setup):
    PATH_model = PATH + "Data_generated/Perturbation_models/Model_files/" + model_type + "/"
    pattern = "*" + '_' + setup + "_" + model_type + "_" + cancer_type + "_" + gene + "*"


    ### TEMPORARY ###
    # PATH_model = "/home/shair/Desktop/Gil/2023_STAMP_recovered_from_drive/Model_files_v3.0/Pathways/GCN/"
    # pattern = "*" + '_' + "LIHC" + "_" + "RTK RAS" + "*"

    model_path = find(pattern, PATH_model)
    if len(model_path) > 1:
        print("error, too many files fit description.")
        exit()
    if model_type == "GCN":
        return torch.load(model_path[0])
    else:
        with open(model_path[0], 'rb') as file:
            Classical_model = pickle.load(file)
        return Classical_model

# subfunction for load_model
# return 'result' containing file name (should be only one) in 'path' with 'pattern' in them
def find(pattern, path):
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result.append(os.path.join(root, name))
    return result

# Load GDSC rnaseq data and parse:
# a. keep only 1st degree
# b. change column names to cell line names
# c. get rid of unnecessary columns
# d. transpose
# e. compute zscore(log(rsem values))
# f. add first degree missing columns as 0 values
def parse_GDSC_rnaseq(first_degree_neighbors, cancer_type):
    data_PATH = PATH+"Data_from_source/GDSC/"

    # read GDSC rnaseq
    GDSC_df = pd.read_csv(data_PATH + 'Cell_line_RMA_proc_basalExp.csv', sep='\t')
    # read GDSC cell line-target-drugs data
    GDSC_cell_target_drug  = pd.read_csv(data_PATH + 'GDSC2_fitted_dose_response_25Feb20.csv').loc[:,
                           ['COSMIC_ID', 'CELL_LINE_NAME', 'TCGA_DESC', 'DRUG_NAME', 'PUTATIVE_TARGET','LN_IC50', 'AUC']].drop_duplicates()
    COSMIC_to_Cell_line_name = GDSC_cell_target_drug.loc[:, ['COSMIC_ID', 'CELL_LINE_NAME']].drop_duplicates()
    COSMIC_to_Cell_line_name.index = COSMIC_to_Cell_line_name['COSMIC_ID']


    # get hugo values in intersect of GDSC and 1st deg neighbors
    in_first_deg_and_GDSC = [x for x in GDSC_df['GENE_SYMBOLS'] if x in first_degree_neighbors]
    # use hugo as index so I can call loc for 1st deg neighbors.
    GDSC_df.index = GDSC_df['GENE_SYMBOLS']
    # keep in GDSC only intersect genes, by the order of the intersect vector
    GDSC_df = GDSC_df.loc[in_first_deg_and_GDSC]

    # replace GDSC_df's columns with cell_line_names, transpose, remove duplicates
    GDSC_rna_cosmic_values = list(set([int(str(col).split(".")[1]) for col in GDSC_df.columns[2:]]))
    GDSC_rna_cosmic_values = [val for val in GDSC_rna_cosmic_values if val in COSMIC_to_Cell_line_name.index]
    COSMIC_to_Cell_line_name = COSMIC_to_Cell_line_name.loc[GDSC_rna_cosmic_values]
    # Transpose, get rid of first 2 columns
    GDSC_df = GDSC_df.iloc[:,2:].transpose()
    GDSC_df.index = [int(str(ind).split(".")[1]) for ind in GDSC_df.index]
    GDSC_df = GDSC_df.loc[GDSC_rna_cosmic_values].drop_duplicates()
    GDSC_df.index = COSMIC_to_Cell_line_name.loc[GDSC_df.index, 'CELL_LINE_NAME']


    # Parse GDSC into z scores similar to TCGA - zscore(log(rna_seq values))
    GDSC_df = np.log(GDSC_df + (1e-10))
    for col in GDSC_df.columns:
        GDSC_df[col] = scipy.stats.zscore(GDSC_df[col])


    # keep only cell line from relevant tissue before normalization
    if cancer_type != "pan_cancer":
        GDSC_cell_line_to_tissue = GDSC_cell_target_drug.loc[:, ["CELL_LINE_NAME", "TCGA_DESC"]].drop_duplicates()
        if cancer_type in ["COAD", "READ"]:
            tissue_logic = GDSC_cell_line_to_tissue['TCGA_DESC'] == "COREAD"
            GDSC_df = GDSC_df.loc[[cosm for cosm in GDSC_df.index if cosm in GDSC_cell_line_to_tissue.loc[tissue_logic,"CELL_LINE_NAME"].values]]
        else:
            tissue_logic = GDSC_cell_line_to_tissue['TCGA_DESC'] == cancer_type
            GDSC_df = GDSC_df.loc[[cosm for cosm in GDSC_df.index if cosm in GDSC_cell_line_to_tissue.loc[tissue_logic,"CELL_LINE_NAME"].values]]


    # add and reorder GDSC columns to fit TCGA
    if model_type == "GCN":
        add_columns = [col for col in gene_model.X.columns if col not in GDSC_df.columns]
    # for cases where model.X.columns have duplicate values (it happens for some reason with EGFR in GBM)
    [GDSC_df.insert(0, col, 0) for col in add_columns]

    GDSC_df = GDSC_df.reindex(columns=first_degree_neighbors)

    # fix (hopefully) a specific bug in EGFR
    # if gene == "EGFR":
    #     GDSC_df.insert(120, 'AK4', 0, allow_duplicates=True)
    return GDSC_df


# Load GDSC rnaseq data and parse:
# a. keep only 1st degree
# b. change column names to cell line names
# c. get rid of unnecessary columns
# d. transpose
# e. compute zscore(log(rsem values))
# f. add first degree missing columns as 0 values
def parse_ENLIGHT_rnaseq(first_degree_neighbors, cancer_type, sorafenib_data = 0):
    data_PATH = PATH+"Data_from_source/ENLIGHT/"

    # read rnaseq
    if cancer_type == "BRCA" or sorafenib_data == 2:
        ENLIGHT_df = pd.read_csv(data_PATH + 'Sorafenib_2.csv', sep=',')
    if cancer_type == "LIHC" or sorafenib_data == 1:
        ENLIGHT_df = pd.read_csv(data_PATH + 'Sorafenib.csv',  sep=',')



    # read GDSC cell line-target-drugs data
    ENLIGHT_cell_target_drug  = pd.read_csv(data_PATH + 'Enlight_drug_response_classification.csv', sep=',')


    # get hugo values in intersect of ENLIGHT and 1st deg neighbors
    in_first_deg_and_ENLIGHT = [x for x in ENLIGHT_df['GENE_SYMBOLS'] if x in first_degree_neighbors]
    # use hugo as index so I can call loc for 1st deg neighbors.
    ENLIGHT_df.index = ENLIGHT_df['GENE_SYMBOLS']
    # keep in ENLIGHT only intersect genes, by the order of the intersect vector
    ENLIGHT_df = ENLIGHT_df.loc[in_first_deg_and_ENLIGHT]

    # Transpose, get rid of first 2 columns
    ENLIGHT_df = ENLIGHT_df.iloc[:,1:].transpose()
    ENLIGHT_df = ENLIGHT_df.drop_duplicates()


    # Parse GDSC into z scores similar to TCGA - zscore(log(rna_seq values))
    ENLIGHT_df = np.log(ENLIGHT_df + (1e-10))
    for col in ENLIGHT_df.columns:
        ENLIGHT_df[col] = scipy.stats.zscore(ENLIGHT_df[col])


    # add and reorder GDSC columns to fit TCGA
    if model_type == "GCN":
        add_columns = [col for col in gene_model.X.columns if col not in ENLIGHT_df.columns]
    else:
        add_columns = [col for col in first_degree_neighbors if col not in ENLIGHT_df.columns]

    # for cases where model.X.columns have duplicate values (it happens for some reason with EGFR in GBM)
    [ENLIGHT_df.insert(0, col, 0) for col in add_columns]

    ENLIGHT_df = ENLIGHT_df.reindex(columns=first_degree_neighbors)

    # fix (hopefully) a specific bug in EGFR
    # if gene == "EGFR":
    #     GDSC_df.insert(120, 'AK4', 0, allow_duplicates=True)
    return ENLIGHT_df


# def get_neighbors_for_classical_model(gene_graphs, gene_model, save_feature_importance=False):
#     neighbors_funcoup, neighbors_humannetv2 = gene_graphs
#     if model_type == "ELR": feature_importance = gene_model.coef_
#     if model_type == "RF": feature_importance = gene_model.feature_importances_
#
#     if len(feature_importance) == len(neighbors_funcoup):
#         # Save feature importance?
#         if save_feature_importance:
#             pd.DataFrame(
#                 {"first degree neighbors": neighbors_funcoup, "feature_importance": feature_importance}
#             ).to_csv(PATH_output + "/Classical_models_feature_importance/" + model_type + "_" + gene + "_" + cancer_type + "_funcoup_feature_importance.csv")
#
#         return pd.Index(neighbors_funcoup).drop_duplicates()
#     if len(feature_importance) == len(neighbors_humannetv2):
#         # Save feature importance?
#         if save_feature_importance:
#             pd.DataFrame(
#                 {"first degree neighbors": neighbors_humannetv2, "feature_importance": feature_importance}
#             ).to_csv(PATH_output + "/Classical_models_feature_importance/" + model_type + "_" + gene + "_" + cancer_type + "_humannetv2_feature_importance.csv")
#         return pd.Index(neighbors_humannetv2).drop_duplicates()
#     print("does not fit graphs. exiting..")
#     return 0


# Example neighbor generation functions
# def funcoup_neighbors(gene):
#     neighbors_funcoup = list(gene_graph_funcoup.first_degree(gene)[0])
#     neighbors_funcoup = [n for n in neighbors_funcoup if n in dataset_df_copy.columns.values]
#
#     # modify adj for neighbors- funcoup
#     # get subgraph for distinct genes
#     neighborhood_funcoup = gene_graph_funcoup.nx_graph.subgraph(neighbors_funcoup).copy()
#
#     # SUBGRAPH NODES ARE NOT ORDERED- ORDER THEM using this line of code.
#     neighbors_funcoup = list(neighborhood_funcoup.nodes)
#     return neighbors_funcoup


# generate funcoup neighbor list for more than one gene together (no dups)
# def get_funcoup_neighbors_of_list(gene_list):
#     new_gene_list = gene_list.copy()
#     for gene in gene_list:
#         new_gene_list.extend(funcoup_neighbors(gene))
#     return list(set(new_gene_list))
# generate HV2 neighbor list for more than one gene together (no dups)



### MAIN ###
if __name__ == '__main__':

    # Parameters
    # save_feature_importance = True
    delay_seconds = 5
    data_to_predict = ["ENLIGHT"] # ["GDSC", "ENLIGHT", "CCLE"]

    # Important genes list.
    Sorafenib_targets = ["KIT", "FLT3", "RET", "PDGFRB"]  # "BRAF", "RAF1", already trained.
    RTK_RAS_dominant_genes = ["EGFR", "ERRFI1", "KRAS", "MET", "NF1", "RASA1"]
    # TP53_pathway_genes = ["TP53", "MDM2", "MDM4", "CDKN2A", "ATM", "CHEK2", "RPS6KA3"]





    # graphs dictionary
    # graph_dict = {"regnet": RegNetGraph, "genemania": GeneManiaGraph,
    #               "humannetv2": HumanNetV2Graph, "funcoup": FunCoupGraph}
    # # load graph once for funcoup
    # gene_graph_funcoup = graph_dict["funcoup"]()

    # Read in data: TCGA
    dataset = TCGADataset()
    # Remove duplicate genes!
    dataset.df = dataset.df.loc[:, ~dataset.df.columns.duplicated()]
    # save a copy of dataset, since it will be modified for cancer_type specific samples
    dataset_df_copy = dataset.df

    # sorafenib_neighbors_funcoup = get_funcoup_neighbors_of_list(Sorafenib_targets)
    # rtk_ras_dominant_neighbors_funcoup = get_funcoup_neighbors_of_list(RTK_RAS_dominant_genes)
    for setup in ["sorafenib_targets", "rtk_ras_dominant"]:
        # Set up gene list and neighbors
        if setup == "sorafenib_targets":
            gene_tumor_type_targets_Sorafenib_project = {
            # Sorafenib targets:
            "BRAF": ["pan_cancer"],
            "RAF1": ["pan_cancer"],
            "PDGFRB": ["pan_cancer"],
            "KIT": ["pan_cancer"],
            "FLT3": ["pan_cancer"],
            "RET": ["pan_cancer"],
            "all_neighbors": ["LIHC", "pan_cancer"]
            }
            gene_list = Sorafenib_targets
            # neighbors_funcoup_setup = sorafenib_neighbors_funcoup
        if setup == "rtk_ras_dominant":
            gene_tumor_type_targets_Sorafenib_project = {
                # RTK RAS dominant genes:
                "EGFR": ["LIHC"],
                "KRAS": ["LIHC"],
                "MET": ["LIHC"],
                "NF1": ["LIHC"],
                "RASA1": ["LIHC"],
                "ERRFI1": ["LIHC"],
                "all_neighbors": ["LIHC", "pan_cancer"]
            }
            gene_list = RTK_RAS_dominant_genes
            # neighbors_funcoup_setup = rtk_ras_dominant_neighbors_funcoup

        for gene in gene_tumor_type_targets_Sorafenib_project.keys():
            # if gene == "all_neighbors":
            #     neighbors_funcoup = list(set(neighbors_funcoup_setup))
            # else:
            #     neighbors_funcoup = list(set(neighbors_funcoup_setup) - set(funcoup_neighbors(gene)))

            # modify adj for neighbors- funcoup
            # get subgraph for distinct genes
            # neighborhood_funcoup = gene_graph_funcoup.nx_graph.subgraph(neighbors_funcoup).copy()
            # get adj for distinct nodes
            # adj_funcoup = np.asarray(nx.to_numpy_matrix(neighborhood_funcoup))
            # SUBGRAPH NODES ARE NOT ORDERED- ORDER THEM using this line of code.
            # neighbors_funcoup = list(neighborhood_funcoup.nodes)


            for model_type in ["GCN", "RF", "ELR"]:
                for cancer_type in gene_tumor_type_targets_Sorafenib_project[gene]:
                    ## Get gene's graph & first degree variables
                    gene_model = load_model(model_type, cancer_type, gene, setup)
                    # 1st degree neighbors
                    if model_type == "GCN":
                        gene_first_degree_neighbors = gene_model.X.columns.drop_duplicates()
                    else:
                        gene_first_degree_neighbors = gene_model.gene_vector

                    for data in ["ENLIGHT"]: # , "GDSC", "CCLE"]:
                        for sorafenib_data in [1, 2]: #, "ELR", "RF"]:

                            if (cancer_type == "LIHC" and sorafenib_data == 2) or (cancer_type == "BRCA" and sorafenib_data == 1):
                                continue

                            if data == "GDSC":
                                # parsre GDSC for model predictions:
                                data_first_deg_z_scored = parse_GDSC_rnaseq(gene_first_degree_neighbors, cancer_type)
                            if data == "ENLIGHT":
                                data_first_deg_z_scored = parse_ENLIGHT_rnaseq(gene_first_degree_neighbors, cancer_type, sorafenib_data = sorafenib_data)
                            # if data == "CCLE":
                                # data_first_deg_z_scored = parse_CCLE_rnaseq(gene_first_degree_neighbors, cancer_type)


                            print("Predicting for setup: ", setup, " for gene: ", gene, " in cancer type: ", cancer_type, " using data: ", data, " with model: ", model_type)
                            # get predictions for data
                            gene_model_predict = gene_model.predict(data_first_deg_z_scored)

                            if model_type == "GCN":
                                data_predict_bool = np.argmax(gene_model_predict, axis=1)
                                data_predict_linear = scipy.special.logit(gene_model_predict[:, 1])

                                # For GCN there may be inf values, replace them with max and min values
                                max_value = max(data_predict_linear[data_predict_linear != math.inf])
                                min_value = min(data_predict_linear[data_predict_linear != -math.inf])
                                data_predict_linear[data_predict_linear == math.inf] = max_value + 1
                                data_predict_linear[data_predict_linear == -math.inf] = min_value - 1
                                # Convert tensor to numpy array
                                data_predict_bool = data_predict_bool.numpy()
                                data_predict_linear = data_predict_linear.numpy()
                            if model_type == "ELR":
                                # Turn classical models' predictions to logistic predictions (use sigmoid function)
                                # Apply the sigmoid function to convert these into probabilities
                                sigmoid = lambda z: 1 / (1 + np.exp(-z))
                                probabilities = sigmoid(gene_model_predict)
                                data_predict_linear = probabilities

                                # Convert probabilities to class predictions (0 or 1) using a threshold of 0.5 on probabilities (after sigmoid)
                                data_predict_bool = (probabilities > 0.5).astype(int)

                            if model_type == "RF":
                                data_predict_bool = gene_model_predict
                                data_predict_linear = scipy.special.logit(gene_model.predict_proba(data_first_deg_z_scored)[:,1])

                            # For all model types:
                            # Stretch linear score between 0 and 1 (minmax)
                            data_predict_linear = (data_predict_linear - np.min(data_predict_linear)) / (
                                        np.max(data_predict_linear) - np.min(data_predict_linear))


                            # create dataframe with model preds for selected data
                            data_predictions = pd.DataFrame(data_first_deg_z_scored.index)
                            data_predictions.index = data_predictions[0]
                            data_predictions = data_predictions.rename(columns={0: data +"_Sample_name"})
                            data_predictions[str(model_type+"_pred")] = data_predict_bool
                            data_predictions[str(model_type+"_linear")] = data_predict_linear
                            data_predictions = data_predictions.iloc[:,1:]
                            if cancer_type == "pan_cancer":
                                data_predictions.to_csv(results_PATH+data+"/data_zscored_pan_"+setup+"_"+data+"_"+gene+"_"+cancer_type+"_"+model_type+"_"+str(sorafenib_data)+"_predictions.csv")
                            else:
                                data_predictions.to_csv(
                                    results_PATH + data + "/data_zscored_pan_" + setup+"_"+data + "_" + gene + "_" + cancer_type + "_" + model_type + "_predictions.csv")

                            if model_type == "GCN":
                                torch.cuda.empty_cache()
                    del gene_model
import pandas as pd
import torch
import os, fnmatch
import numpy as np
from data.datasets import TCGADataset
from matplotlib import pyplot as plt
import scipy
import math
import pickle

PATH = "/home/shair/Desktop/Gil/2024_STAMP_Explainatory_Translational_Sorafenib/"


### Functions

# returns model of model_type, for cancer_type & gene combination
def load_model(model_type, cancer_type, gene):
    PATH_model = PATH + "STAMP_models/Model_files/" + model_type + "/"
    pattern = "*"+'_'+cancer_type + "_" + gene+"*"
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
        ENLIGHT_df = pd.read_csv(data_PATH + 'sorafenib_2.csv', sep='\t')
    if cancer_type == "LIHC" or sorafenib_data == 1:
        ENLIGHT_df = pd.read_csv(data_PATH + 'sorafenib.csv', sep='\t')



    # read GDSC cell line-target-drugs data
    ENLIGHT_cell_target_drug  = pd.read_csv(data_PATH + 'Enlight_drug_response_classification.csv')


    # get hugo values in intersect of ENLIGHT and 1st deg neighbors
    in_first_deg_and_GDSC = [x for x in ENLIGHT_df['GENE_SYMBOLS'] if x in first_degree_neighbors]
    # use hugo as index so I can call loc for 1st deg neighbors.
    ENLIGHT_df.index = ENLIGHT_df['GENE_SYMBOLS']
    # keep in ENLIGHT only intersect genes, by the order of the intersect vector
    ENLIGHT_df = ENLIGHT_df.loc[in_first_deg_and_GDSC]


### todo: Got to this point with this function for ENLIGHT 16.12.2024


    # Transpose, get rid of first 2 columns
    ENLIGHT_df = ENLIGHT_df.iloc[:,2:].transpose()
    ENLIGHT_df.index = [int(str(ind).split(".")[1]) for ind in ENLIGHT_df.index]
    ENLIGHT_df = ENLIGHT_df.drop_duplicates()
    ENLIGHT_df.index = ENLIGHT_df.


    # Parse GDSC into z scores similar to TCGA - zscore(log(rna_seq values))
    ENLIGHT_df = np.log(ENLIGHT_df + (1e-10))
    for col in ENLIGHT_df.columns:
        ENLIGHT_df[col] = scipy.stats.zscore(ENLIGHT_df[col])


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
    add_columns = [col for col in gene_model.X.columns if col not in GDSC_df.columns]
    # for cases where model.X.columns have duplicate values (it happens for some reason with EGFR in GBM)
    [GDSC_df.insert(0, col, 0) for col in add_columns]

    GDSC_df = GDSC_df.reindex(columns=first_degree_neighbors)

    # fix (hopefully) a specific bug in EGFR
    # if gene == "EGFR":
    #     GDSC_df.insert(120, 'AK4', 0, allow_duplicates=True)
    return GDSC_df


### MAIN ###
if __name__ == '__main__':
    # paths
    results_PATH = PATH+"Data_generated/STAMP_predictions/"


    delay_seconds = 5

    data_to_predict = ["ENLIGHT"] # ["GDSC", "ENLIGHT", "CCLE"]

    genes_cancer_types = {
    "RTK RAS": ["pan_cancer", "LIHC", "BRCA", "LUAD", "LUSC", "HNSC"]
    # "BRAF": ["pan_cancer", "LUAD"],
    # "EGFR": ["pan_cancer", "LIHC", "BRCA"],
    # "KRAS": ["pan_cancer", "LIHC", "BRCA"],
    # "MET": ["pan_cancer", "LIHC", "BRCA"],
    # "NF1": ["pan_cancer", "LIHC", "BRCA"],
    # "RASA1": ["pan_cancer", "LIHC", "BRCA"],
    # "ERRFI1": ["pan_cancer", "LIHC", "BRCA"],
    # "FGFR1": ["pan_cancer", "BRCA", "LUSC", "HNSC"]
    }

    for data in data_to_predict:
        for gene in genes_cancer_types.keys():
            for cancer_type in genes_cancer_types[gene]:
                for model_type in ["GCN", "ELR", "RF"]:
                    # get model
                    gene_model = load_model(model_type, cancer_type, gene)
                    #ToDo: check if this (all following lines) works for ELR, RF

                    # 1st degree neighbors
                    gene_first_degree_neighbors = gene_model.X.columns.drop_duplicates()

                    if data == "GDSC":
                        # parsre GDSC for model predictions:
                        data_first_deg_z_scored = parse_GDSC_rnaseq(gene_first_degree_neighbors, cancer_type)
                    if data == "ENLIGHT":
                        if cancer_type == "pan_cancer":
                            # Change sorafenib_data to 2 if you want Breast Data
                            data_first_deg_z_scored = parse_ENLIGHT_rnaseq(gene_first_degree_neighbors, cancer_type, sorafenib_data = 1)
                        else:
                            data_first_deg_z_scored = parse_ENLIGHT_rnaseq(gene_first_degree_neighbors, cancer_type,
                                                                           sorafenib_data=0)

                    # get predictions for GDSC
                    gene_model_predict = gene_model.predict(data_first_deg_z_scored)
                    data_predict_bool = np.argmax(gene_model_predict, axis=1)
                    # calculate linear score, avoid infinity values
                    data_predict_linear = scipy.special.logit(gene_model_predict[:, 1])
                    max_value = max(data_predict_linear[data_predict_linear != math.inf])
                    min_value = min(data_predict_linear[data_predict_linear != -math.inf])
                    data_predict_linear[data_predict_linear == math.inf] = max_value + 1
                    data_predict_linear[data_predict_linear == -math.inf] = min_value - 1

                    # create dataframe with GCN preds for selected data
                    data_predictions = pd.DataFrame(data_first_deg_z_scored.index)
                    data_predictions.index = data_predictions["Sample_name"]
                    data_predictions = data_predictions.rename(columns={0: data +"_Sample_name"})
                    data_predictions["GCN_pred"] = data_predict_bool.numpy()
                    data_predictions["GCN_linear"] = data_predict_linear.numpy()
                    data_predictions = data_predictions.iloc[:,1:]

                    data_predictions.to_csv(results_PATH+data+"/data_zscored_pan_"+data+"_"+gene+"_"+cancer_type+"_"+model_type+"_predictions.csv")

                    del gene_model
                    if model_type == "GCN":
                        torch.cuda.empty_cache()

                    # # load GDSC cosmic to cell line name data (drugs effect data)
                    # GDSC_cosmic_to_cell_line = pd.read_csv(data_PATH + 'GDSC2_fitted_dose_response_25Feb20.csv').loc[:,
                    #                            ['COSMIC_ID', 'CELL_LINE_NAME', 'TCGA_DESC', 'DRUG_NAME', 'PUTATIVE_TARGET','LN_IC50', 'AUC']].drop_duplicates()
                    #
                    # Dabrafenib_IC_50 = GDSC_cosmic_to_cell_line.loc[GDSC_cosmic_to_cell_line['DRUG_NAME'] == "PLX-4720", 'LN_IC50']
                    # Dabrafenib_IC_50 = Dabrafenib_IC_50.loc[[val for val in Dabrafenib_IC_50.index if val in GDSC_predictions.index]]
                    # GDSC_IC50 = GDSC_predictions.loc[Dabrafenib_IC_50.index]
                    # GDSC_IC50['Dabrafenib_IC50'] = Dabrafenib_IC_50.loc[GDSC_IC50.index]
                    # plt.scatter(GDSC_IC50['GCN_linear'], GDSC_IC50['Dabrafenib_IC50'])
import pandas as pd
import torch
import os, fnmatch
import numpy as np
from data.datasets import TCGADataset
from matplotlib import pyplot as plt
import scipy
import math


### Functions

# returns model of model_type, using data_type, for cancer_type & gene combination
def load_model(data_type, model_type, cancer_type, gene):
    PATH_model = "/home/gil/Desktop/GCN/GCN_analysis/2.models/" + data_type + "/" + model_type + "/"
    pattern = "*"+'_'+cancer_type + "_" + gene+"*"
    model_path = find(pattern, PATH_model)
    if len(model_path) > 1:
        print("error, too many files fit description.")
        exit()
    return torch.load(model_path[0])

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
    data_PATH = "/home/gil/Desktop/GCN/datasets/Data_downloaded_directly_from_source/"

    # read GDSC rnaseq
    GDSC_df = pd.read_csv(data_PATH + 'GDSC_Cell_line_RMA_proc_basalExp.csv')
    # read GDSC cell line-target-drugs data data
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


### MAIN ###
if __name__ == '__main__':
    # paths
    PATH_GCN = "/home/gil/Desktop/GCN/"
    data_PATH = "/home/gil/Desktop/GCN/datasets/Data_downloaded_directly_from_source/"
    results_PATH = "/home/gil/Desktop/GCN/GCN_analysis/10.CCLE/GDSC_models_3.0/"

    # define model/gene/cancer_type/drug to investigate
    data_type = "Pathways"
    model_type = "GCN"
    delay_seconds = 5
    genes_cancer_types = {"PI3K": ["LUAD", "LUSC", "BRCA"]}#"TP53":["BRCA", "LUAD", "LGG"]}#"EGFR":["GBM"], "ERBB2":["BRCA"], "KRAS":["LUAD"], "BRAF":["SKCM", "THCA"]} #, "COAD", "READ", "PAAD"], "RTK RAS":["BRCA", "UCEC", "THCA", "LGG", "LUSC", "HNSC", "LUAD", "SKCM", "BLCA", "COAD", "STAD", "CESC", "SARC", "ESCA", "PAAD", "GBM", "READ", "OV", "LAML", "TGCT", "PCPG", "MESO", "CHOL"]}

    for gene in genes_cancer_types.keys():
        for cancer_type in genes_cancer_types[gene]:

            # get model
            gene_model = load_model(data_type, model_type, cancer_type, gene)
            # 1st degree neighbors
            gene_first_degree_neighbors = gene_model.X.columns.drop_duplicates()

            # parsre GDSC for model predictions:
            GDSC_first_deg_z_scored = parse_GDSC_rnaseq(gene_first_degree_neighbors, cancer_type)

            # get predictions for GDSC
            gene_model_predict = gene_model.predict(GDSC_first_deg_z_scored)
            GDSC_predict_bool = np.argmax(gene_model_predict, axis=1)
            # calculate linear score, avoid infinity values
            GDSC_predict_linear = scipy.special.logit(gene_model_predict[:, 1])
            max_value = max(GDSC_predict_linear[GDSC_predict_linear != math.inf])
            min_value = min(GDSC_predict_linear[GDSC_predict_linear != -math.inf])
            GDSC_predict_linear[GDSC_predict_linear == math.inf] = max_value + 1
            GDSC_predict_linear[GDSC_predict_linear == -math.inf] = min_value - 1

            # create GDSC dataframe with GCN preds
            GDSC_predictions = pd.DataFrame(GDSC_first_deg_z_scored.index)
            GDSC_predictions.index = GDSC_predictions["CELL_LINE_NAME"]
            GDSC_predictions = GDSC_predictions.rename(columns={0: "GDSC Cell Line Name"})
            GDSC_predictions["GCN_pred"] = GDSC_predict_bool.numpy()
            GDSC_predictions["GCN_linear"] = GDSC_predict_linear.numpy()
            GDSC_predictions = GDSC_predictions.iloc[:,1:]

            GDSC_predictions.to_csv(results_PATH+"GDSC_cell_lines_zscored_pan_GDSC_"+gene+"_"+cancer_type+"_GCN_predictions.csv")

            del gene_model
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
PATH = "~/Desktop/GCN_from_scratch/"

import pandas as pd
import numpy as np
import sys
import csv
import torch

PATH = "/home/shair/Desktop/Gil/2023_STAMP_recovered_from_drive/"


# input: read sample ids by cancer type from cbioportal clinical table
# read table of sample IDs mutated in 'gene' and create label accordingly
# data can be 'firehose_pan_cancer' or 'Breast_cancer'
# gene currently can be TP53
# returns: Sample ID and label (0, 1) lists.
def get_samples_and_labels_by_cancer_type_gene(cancer_type, data, gene, pathway_type="PATHWAY"):
    if data == "BRCA_GENE":

        with open(PATH +
                    '/datasets/Data_downloaded_directly_from_source/' + gene + '_pan_cancer_label_from_cbioportal.csv'
                    ) as csv_file:
                    csv_reader = csv.DictReader(csv_file, delimiter=',')

                    if cancer_type == "pan_cancer":
                        Sample_IDs = []
                        labels = []
                        for row in csv_reader:
                            Sample_IDs.append(row['Sample.ID'])
                            labels.append(int(row[gene]))
                    else:
                        Sample_IDs = []
                        labels = []
                        for row in csv_reader:
                            if row['tumor_type'] == cancer_type:
                                Sample_IDs.append(row['Sample.ID'])
                                labels.append(int(row[gene]))

    elif data == "combined_cbioportal_TCGAbiolinks":
        with open(PATH +
                  '/datasets/Data_downloaded_directly_from_source/TCGAbiolinks_cbioportal_combined_profiled_sample_IDs_with_labels.csv'
                  ) as csv_file:
            csv_reader = csv.DictReader(csv_file, delimiter=',')
            if cancer_type == "pan_cancer":
                Sample_IDs = []
                labels = []
                for row in csv_reader:
                    Sample_IDs.append(row['Sample.ID'])
                    labels.append(int(row[gene]))
            else:
                Sample_IDs = []
                labels = []
                for row in csv_reader:
                    if row['tumor_type'] == cancer_type:
                        Sample_IDs.append(row['Sample.ID'])
                        labels.append(int(row[gene]))
    else:
        if data == "pathway_cell_paper":
            # get Sample IDs according to tumor_type
            with open(PATH +
                      '/datasets/Data_downloaded_directly_from_source/TCGAbiolinks_cbioportal_combined_profiled_sample_IDs_with_labels.csv'
                      ) as csv_file:
                csv_reader = csv.DictReader(csv_file, delimiter=',')
                if cancer_type == "pan_cancer":
                    Sample_IDs_tumor_type = []
                    for row in csv_reader:
                        Sample_IDs_tumor_type.append(row['Sample.ID'])
                else:
                    Sample_IDs_tumor_type = []
                    for row in csv_reader:
                        if row['tumor_type'] == cancer_type:
                            Sample_IDs_tumor_type.append(row['Sample.ID'])

            # now get the same Sample IDs with their lables
            with open(
                    PATH + '/datasets/Data_downloaded_directly_from_source/Oncogenic_signaling_pathways_' +
                    pathway_type + '_level.csv') as csv_file:
                csv_reader = csv.DictReader(csv_file, delimiter=',')
                Sample_IDs = []
                labels = []
                for row in csv_reader:
                    if row['SAMPLE_BARCODE'] in Sample_IDs_tumor_type:
                        # skip NAs or any other form that is not the binary 0/1
                        if(row[gene] != '0' and row[gene] != '1'):
                            continue
                        else:
                            Sample_IDs.append(row['SAMPLE_BARCODE'])
                            labels.append(int(row[gene]))

        else:
            # 1. get sample IDs according to data and cancer type
            with open(
                    PATH + '/datasets/Data_downloaded_directly_from_source/cbioportal_TCGA_' + data + '_all_samples1.0.tsv') as csv_file:
                csv_reader = csv.DictReader(csv_file, delimiter='\t')
                # 1a. for pan cancer (firehose), read all sample IDs
                if cancer_type == "pan_cancer":
                    Sample_IDs = [row['Sample ID'] for row in csv_reader]
                else:
                    # 1b. for BRCA data, read all sample IDs and remove duplicates.
                    if data == "breast_cancer":
                        Sample_IDs = [row['Sample ID'] for row in csv_reader]
                        # for Breast_cancer, also remove duplicates.
                        Sample_IDs = list(dict.fromkeys(Sample_IDs))
                    # 1c. for firehose, specific cancer_types, read only if Study ID fits cancer_type.
                    else:
                        # cancer_type = cancer_type.lower() + "_tcga"
                        Sample_IDs = [row['Sample ID'] for row in csv_reader if row['Study ID'] == cancer_type]

            # 2. create labels for Sample IDs list. use data and gene to read the correct file.
            with open(
                    PATH + '/datasets/Data_downloaded_directly_from_source/cbioportal_TCGA_' + data + '_' + gene + '_mutated_samples1.0.tsv') as csv_file:
                csv_reader = csv.DictReader(csv_file, delimiter='\t')
                # 2a. read mutated sample IDs
                Sample_IDs_mutated = [row['Sample ID'] for row in csv_reader]
            # 2b. create labels by the order of Sample_IDs
            labels = [1 if Sample_ID in Sample_IDs_mutated else 0 for Sample_ID in Sample_IDs]
    return Sample_IDs, labels


def get_firehose_cancer_types():
    data = "firehose_pan_cancer"
    with open(
            PATH + '/datasets/Data_downloaded_directly_from_source/cbioportal_TCGA_' + data + '_all_samples1.0.tsv') as csv_file:
        csv_reader = csv.DictReader(csv_file, delimiter='\t')
        Study_IDs = [row['Study ID'] for row in csv_reader]
    return list(dict.fromkeys(Study_IDs))

def get_cell_paper_pathway_list(pathway_type="PATHWAY"):
    with open(
            PATH + '/datasets/Data_downloaded_directly_from_source/Oncogenic_signaling_pathways_' +
            pathway_type + '_level.csv') as csv_file:
            csv_reader = csv.DictReader(csv_file, delimiter=',')
            dict_from_csv = dict(list(csv_reader)[0])

            # making a list from the keys of the dict
            list_of_column_names = list(dict_from_csv.keys())
            return list_of_column_names[1:len(list_of_column_names)]

def get_cbioportal_TCGAbiolinks_cancer_types():
    with open(PATH +
              '/datasets/Data_downloaded_directly_from_source/TCGAbiolinks_cbioportal_combined_profiled_sample_IDs_with_labels.csv'
              ) as csv_file:
        csv_reader = csv.DictReader(csv_file, delimiter=',')
        Study_IDs = [row['tumor_type'] for row in csv_reader]
    return(list(dict.fromkeys(Study_IDs)))

def save_predictions_table(X_train, y_train, X_val, y_val, X_test, y_test, model, model_type):

    # get scores from model
    if model_type == "RF":
        y_train_hat = model.predict_proba(X_train)[:, 1]
        y_val_hat = model.predict_proba(X_val)[:, 1]
        y_test_hat = model.predict_proba(X_test)[:, 1]
    else:
        y_train_hat = model.predict(X_train)
        y_val_hat = model.predict(X_val)
        y_test_hat = model.predict(X_test)

    # generate dataframe for saving
    if model_type == "GCN" or model_type == "MLP" or model_type == "BIG_GCN":
        train_df = pd.DataFrame(
            {"Sample ID": X_train.index.values, "Set": "train", "label": y_train,
             "prediction": np.argmax(y_train_hat, axis=1),
             "pred_score": y_train_hat[:, 1]}
        )
        val_df = pd.DataFrame(
            {"Sample ID": X_val.index.values, "Set": "val", "label": y_val,
             "prediction": np.argmax(y_val_hat, axis=1),
             "pred_score": y_val_hat[:, 1]}
        )
        test_df = pd.DataFrame(
            {"Sample ID": X_test.index.values, "Set": "test", "label": y_test,
             "prediction": np.argmax(y_test_hat, axis=1),
             "pred_score": y_test_hat[:, 1]}
        )
    else:
        train_df = pd.DataFrame(
            {"Sample ID": X_train.index.values, "Set": "train", "label": y_train,
             "prediction": list(map(int, y_train_hat > 0.5)),
             "pred_score": y_train_hat}
        )
        val_df = pd.DataFrame(
            {"Sample ID": X_val.index.values, "Set": "val", "label": y_val,
             "prediction": list(map(int, y_val_hat > 0.5)),
             "pred_score": y_val_hat}
        )
        test_df = pd.DataFrame(
            {"Sample ID": X_test.index.values, "Set": "test", "label": y_test,
             "prediction": list(map(int, y_test_hat > 0.5)),
             "pred_score": y_test_hat}
        )
    frames = [train_df, val_df, test_df]
    pred_df = pd.concat(frames)
    return pred_df

def get_TCGAbiolinks_expression_column_names():
    # correct columns for expression file
    columns = []
    with open(
            PATH + 'datasets/Data_downloaded_directly_from_source/RNASeq_33_TCGA_projects_downloaded_with_TCGAbiolinks_2021_03_01.csv') as csv_file:
        csv_reader = csv.DictReader(csv_file, delimiter=',')
        for row in csv_reader:
            for name in row:
                columns.append(name.split("|")[0])
            break
    return columns

def get_TCGAbiolinks_expression_Sample_IDs_and_cancer_type():
    row_names_project_id = pd.read_csv(
        PATH + 'datasets/Data_downloaded_directly_from_source/RNASeq_33_TCGA_projects_downloaded_with_TCGAbiolinks_2021_03_01.csv',
        usecols=["Sample_ID", "project"])
    row_names = row_names_project_id["Sample_ID"].tolist()
    row_names = [name.split('_')[2][:15] for name in row_names]
    row_names_project_id["Sample_ID"] = row_names
    # remove 9 duplicates
    row_names_project_id = row_names_project_id.drop_duplicates()
    return row_names_project_id

def get_genes_selection_set(selection_type):
    if selection_type == "deseq":
        gene_set = pd.read_csv(PATH + 'datasets/gene_selection_sets/DeSeq_genes_selection_300_set.csv')
    if selection_type == "mutsigdb":
        gene_set = pd.read_csv(PATH + 'datasets/gene_selection_sets/MutsigDB_FISCHER_DIRECT_P53_TARGETS_META_ANALYSIS_311_genes_set.csv')
    return gene_set.iloc[:,0].values.tolist()
#%%
# Goal: Identify neighbor genes of each target, check the overlap of all targets. check the same for dominant genes.
# Run the same analysis for MDM2 and the TP53 pathway players. Is there uniqueness for any of them or is the overlap close to 100%?

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
PATH_save_results = '/home/shair/Desktop/Gil/2024_STAMP_Explainatory_Translational_Sorafenib/Data_generated/Perturbation_models/'

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


# Redefine randomforestclassifier + elasticnet, so they can store the genes vector (neighbors used as features by the model, in the right order)
class RandomForest_with_Gene_Vector(RandomForestClassifier):
    def __init__(self, *args, gene_vector=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.gene_vector = gene_vector  # Add the gene_vector attribute

    def set_gene_neighbors(self, vector):
        self.gene_vector = vector  # Method to set the model_vector after training

    def get_gene_vector(self):
        return self.gene_vector  # Method to retrieve the model_vector
class ElasticNet_with_Gene_Vector(ElasticNet):
    def __init__(self, *args, gene_vector=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.gene_vector = gene_vector  # Add the model_vector attribute

    def set_gene_neighbors(self, vector):
        self.gene_vector = vector  # Method to set the model_vector after training

    def get_gene_vector(self):
        return self.gene_vector  # Method to retrieve the model_vector



# Example neighbor generation functions
def funcoup_neighbors(gene):
    neighbors_funcoup = list(gene_graph_funcoup.first_degree(gene)[0])
    neighbors_funcoup = [n for n in neighbors_funcoup if n in dataset_df_copy.columns.values]

    # modify adj for neighbors- funcoup
    # get subgraph for distinct genes
    neighborhood_funcoup = gene_graph_funcoup.nx_graph.subgraph(neighbors_funcoup).copy()

    # SUBGRAPH NODES ARE NOT ORDERED- ORDER THEM using this line of code.
    neighbors_funcoup = list(neighborhood_funcoup.nodes)
    return neighbors_funcoup

def humannetv2_neighbors(gene):
    # adj_humannetv2 = gene_graph_humannetv2.adj()
    neighbors_HV2 = list(gene_graph_humannetv2.first_degree(gene)[0])
    neighbors_HV2 = [n for n in neighbors_HV2 if n in dataset_df_copy.columns.values]

    # modify adj for neighbors- humannet
    # get subgraph for distinct genes
    neighborhood_HV2 = gene_graph_humannetv2.nx_graph.subgraph(neighbors_HV2).copy()

    # SUBGRAPH NODES ARE NOT ORDERED- ORDER THEM using this line of code.
    neighbors_HV2 = list(neighborhood_HV2.nodes)

    return neighbors_HV2

# generate funcoup neighbor list for more than one gene together (no dups)
def get_funcoup_neighbors_of_list(gene_list):
    new_gene_list = gene_list.copy()
    for gene in gene_list:
        new_gene_list.extend(funcoup_neighbors(gene))
    return list(set(new_gene_list))
# generate HV2 neighbor list for more than one gene together (no dups)
def get_HV2_neighbors_of_list(gene_list):
    new_gene_list = gene_list.copy()
    for gene in gene_list:
        new_gene_list.extend(humannetv2_neighbors(gene))
    return list(set(new_gene_list))


#### MAIN ####



# Important genes list.
Sorafenib_targets = [ "KIT", "FLT3", "RET", "PDGFRB"] # "BRAF", "RAF1", already trained.
RTK_RAS_dominant_genes = ["EGFR", "ERRFI1", "KRAS", "MET", "NF1", "RASA1"]
# TP53_pathway_genes = ["TP53", "MDM2", "MDM4", "CDKN2A", "ATM", "CHEK2", "RPS6KA3"]

seed = 1234
cuda = torch.cuda.is_available()
# tuning
data = "pathway_cell_paper"
pathway_type = "PATHWAY" # can also be "GENE" or "ALTERATION"
dropout = False
agg = "hierarchy"
embedding = 30


gene_tumor_type_targets_Sorafenib_project = {
    # RTK RAS dominant genes:
    "EGFR": ["LIHC"],
    "KRAS": ["LIHC"],
    "MET": ["LIHC"],
    "NF1": ["LIHC"],
    "RASA1": ["LIHC"],
    "ERRFI1": ["LIHC"],

    # Sorafenib targets:
    "BRAF": ["pan_cancer"],
    "RAF1": ["pan_cancer"],
    "PDGFRB": ["pan_cancer"],
    "KIT": ["pan_cancer"],
    "FLT3": ["pan_cancer"],
    "RET": ["pan_cancer"],

    # all
    "all_neighbors": ["LIHC", "pan_cancer"]
}



sorafenib_neighbors_funcoup = get_funcoup_neighbors_of_list(Sorafenib_targets)
sorafenib_neighbors_HV2 = get_HV2_neighbors_of_list(Sorafenib_targets)
rtk_ras_dominant_neighbors_funcoup = get_funcoup_neighbors_of_list(RTK_RAS_dominant_genes)
rtk_ras_dominant_neighbors_HV2 = get_HV2_neighbors_of_list(RTK_RAS_dominant_genes)


best_models = []
results = []

for setup in ["sorafenib_targets", "rtk_ras_dominant"]:
    # Set up gene list and neighbors
    if setup == "sorafenib_targets":
        gene_list = Sorafenib_targets + ["all_neighbors"]
        neighbors_funcoup_setup = sorafenib_neighbors_funcoup
        neighbors_HV2_setup = sorafenib_neighbors_HV2
    if setup == "rtk_ras_dominant":
        gene_list = RTK_RAS_dominant_genes + ["all_neighbors"]
        neighbors_funcoup_setup = rtk_ras_dominant_neighbors_funcoup
        neighbors_HV2_setup = rtk_ras_dominant_neighbors_HV2

    ##todo: use this line if you wish running a model on all RTK RAS/Sorafenib neighbors
    # gene_list = ["all_neighbors"]


    for gene in gene_list:
            if gene == "all_neighbors":
                neighbors_funcoup = list(set(neighbors_funcoup_setup))
                neighbors_HV2 = list(set(neighbors_HV2_setup))
            else:
                neighbors_funcoup = list(set(neighbors_funcoup_setup) - set(funcoup_neighbors(gene)))
                neighbors_HV2 = list(set(neighbors_HV2_setup) - set(humannetv2_neighbors(gene)))

            # modify adj for neighbors- funcoup
            # get subgraph for distinct genes
            neighborhood_funcoup = gene_graph_funcoup.nx_graph.subgraph(neighbors_funcoup).copy()
            # get adj for distinct nodes
            adj_funcoup = np.asarray(nx.to_numpy_matrix(neighborhood_funcoup))
            # SUBGRAPH NODES ARE NOT ORDERED- ORDER THEM using this line of code.
            neighbors_funcoup = list(neighborhood_funcoup.nodes)

            # modify adj for neighbors- humannet
            # get subgraph for distinct genes
            neighborhood_HV2 = gene_graph_humannetv2.nx_graph.subgraph(neighbors_HV2).copy()
            # get adj for distinct nodes
            adj_HV2 = np.asarray(nx.to_numpy_matrix(neighborhood_HV2))
            # SUBGRAPH NODES ARE NOT ORDERED- ORDER THEM using this line of code.
            neighbors_HV2 = list(neighborhood_HV2.nodes)

            for cancer_type in gene_tumor_type_targets_Sorafenib_project[gene]:

                Sample_IDs, labels = Parse.get_samples_and_labels_by_cancer_type_gene(cancer_type, data, "RTK RAS",
                                                                                      pathway_type)
                num_mutated = sum(labels)
                n_samples = len(Sample_IDs)
                train_n_samples = int(len(Sample_IDs) * 0.6)

                labels_df = pd.DataFrame(labels, index=Sample_IDs, columns=['label'])
                labels_df = labels_df.sample(frac=1)
                labels_df = labels_df.loc[labels_df.index.isin(dataset_df_copy.T.columns), :]
                dataset.df = dataset_df_copy.loc[labels_df.index.values, :]
                dataset.df = dataset.df.reindex(labels_df.index.values)
                dataset.sample_names = labels_df.index.values.tolist()
                dataset.labels = labels_df['label'].values.tolist()

                # Define train and test sets
                train_size = int(len(dataset.labels) * 0.6)
                val_test_size = len(dataset.labels) - train_size
                X_train, X_val_test, y_train, y_val_test = sklearn.model_selection.train_test_split(dataset.df,
                                                                                                    dataset.labels,
                                                                                                    stratify=dataset.labels,
                                                                                                    train_size=train_size,
                                                                                                    test_size=val_test_size,
                                                                                                    random_state=seed)
                val_size = int(len(dataset.labels) * 0.2)
                test_size = val_test_size - val_size
                X_val, X_test, y_val, y_test = sklearn.model_selection.train_test_split(X_val_test, y_val_test,
                                                                                        stratify=y_val_test,
                                                                                        train_size=val_size,
                                                                                        test_size=test_size,
                                                                                        random_state=seed)

                X_train_copy = X_train
                X_val_copy = X_val
                X_test_copy = X_test

                # ### GCN ###
                # for model_type in ["GCN"]:
                #     # restart best models for every model&cancer_type
                #     best_auc = 0
                #     best_auc_model = None
                #     for graph in ["funcoup", "humannetv2"]:
                #         if graph == "funcoup":
                #             gene_graph = gene_graph_funcoup
                #             adj = adj_funcoup
                #         else:
                #             gene_graph = gene_graph_humannetv2
                #             adj = adj_HV2
                #         for num_layer in [1, 2, 3]:
                #             if num_layer > 1:
                #                 embedding = 40
                #             if num_layer > 3:
                #                 embedding = 50
                #             for channels in [32, 40, 48]:
                #                 for batch_size in [4, 8, 12]:
                #                     for lr in [1e-5, 1e-4, 1e-3]:
                #                         if cancer_type == "pan_cancer":
                #                             lr = lr * 0.01
                #                         is_first_degree = True
                #                         model = GCN(column_names=dataset.df.columns, name="GCN_emb32_agg_hierarchy",
                #                                     cuda=True,
                #                                     aggregation=agg, agg_reduce=2, lr=lr,
                #                                     num_layer=num_layer, channels=channels, embedding=embedding,
                #                                     batch_size=batch_size,
                #                                     dropout=dropout, verbose=False, seed=seed)
                #
                #                         experiment = {
                #                             "setup": setup,
                #                             "gene": gene,
                #                             "num_mutated": num_mutated,
                #                             "model": model_type,
                #                             "graph": graph,
                #                             "gene_selection": "NA",
                #                             "cancer type": cancer_type,
                #                             "num_layer": num_layer,
                #                             "channels": channels,
                #                             "batch": batch_size,
                #                             "lr": lr,
                #                             "ratio": "NA",
                #                             "dropout": dropout,
                #                             "aggregation": agg,
                #                             "criterion": "NA",
                #                             "max_features": "NA",
                #                             "n_estimator": "NA",
                #                         }
                #
                #                         # Training
                #
                #                         if is_first_degree:
                #                             if graph == "funcoup":
                #                                 neighbors = neighbors_funcoup
                #                             else:
                #                                 neighbors = neighbors_HV2
                #                             X_train = X_train_copy.loc[:, neighbors].copy()
                #                             X_val = X_val_copy.loc[:, neighbors].copy()
                #                             X_test = X_test_copy.loc[:, neighbors].copy()
                #                         else:
                #                             X_train = X_train_copy.copy()
                #                             X_val = X_val_copy.copy()
                #                             X_test = X_test_copy.copy()
                #
                #                         try:
                #                             with warnings.catch_warnings():
                #                                 warnings.simplefilter("ignore")
                #                                 if model_type == "GCN":
                #                                     model.fit(X_train, y_train, adj)
                #                                 if model_type == "MLP" or model_type == "LR":
                #                                     model.fit(X_train, y_train)
                #                                 model.eval()
                #                                 with torch.no_grad():
                #                                     y_val_hat = model.predict(X_val)
                #                             val_auc = sklearn.metrics.roc_auc_score(y_val, y_val_hat[:, 1])
                #                             val_acc = sklearn.metrics.accuracy_score(y_val,
                #                                                                      np.argmax(y_val_hat, axis=1))
                #                             val_prec = sklearn.metrics.precision_score(y_val,
                #                                                                        np.argmax(y_val_hat, axis=1))
                #                             val_recall = sklearn.metrics.recall_score(y_val,
                #                                                                       np.argmax(y_val_hat, axis=1))
                #
                #                             print("gene: ", gene, "model: ", model_type, ", cancer_type: ",
                #                                   cancer_type, ", graph: ", graph, ", layers: ", num_layer,
                #                                   ", channels: ", channels,
                #                                   ", batch_size: ", batch_size, " lr: ", lr, ", dropout: ", dropout,
                #                                   ", val auc:", val_auc, ", val acc: ", val_acc, ", precision: ",
                #                                   val_prec, ", recall: ", val_recall)
                #
                #                             experiment["val_auc"] = val_auc
                #                             experiment["val_acc"] = val_acc
                #                             experiment["val_prec"] = val_prec
                #                             experiment["val_recall"] = val_recall
                #                             results.append(experiment)
                #                             if val_auc > best_auc:
                #                                 best_auc = val_auc
                #                                 best_auc_model = model
                #                                 best_auc_X_train = X_train
                #                                 best_auc_X_val = X_val
                #                                 best_auc_X_test = X_test
                #                                 best_auc_exp = experiment.copy()
                #
                #                             model.best_model = None  # cleanup
                #                             del model
                #                             torch.cuda.empty_cache()
                #                         except Exception:
                #                             tb = traceback.format_exc()
                #                             experiment['val_recall'] = tb
                #                             print(tb)
                #
                #     # Pred_table AUC
                #     pred_df_auc = Parse.save_predictions_table(best_auc_X_train, y_train,
                #                                                best_auc_X_val, y_val, best_auc_X_test, y_test,
                #                                                best_auc_model, model_type)
                #     pred_df_auc.to_csv(
                #         PATH_save_results + "Prediction_tables/" + model_type + "/pred_table_Perturbations_RTK_RAS_label" + "_" + setup + "_" + model_type + "_" +
                #         cancer_type + "_" + gene + "_val_AUC_" + str(best_auc) + ".csv"
                #     )
                #
                #     # save best Models
                #     torch.save(best_auc_model,
                #                PATH_save_results + "Model_files/" + model_type + "/best_model_Perturbations_RTK_RAS_label"+ "_" + setup + "_" + model_type + "_" +
                #                cancer_type + "_" + gene + "_val_AUC_" + str(best_auc) + ".pt"
                #                )
                #
                #     ### run best model on Test ###
                #     # run prediction on test set and produce scores
                #     with torch.no_grad():
                #         y_test_hat = best_auc_model.predict(best_auc_X_test)
                #     test_auc = sklearn.metrics.roc_auc_score(y_test, y_test_hat[:, 1])
                #     test_acc = sklearn.metrics.accuracy_score(y_test, np.argmax(y_test_hat, axis=1))
                #     test_prec = sklearn.metrics.precision_score(y_test, np.argmax(y_test_hat, axis=1))
                #     test_recall = sklearn.metrics.recall_score(y_test, np.argmax(y_test_hat, axis=1))
                #
                #     best_auc_exp['test_auc'] = test_auc
                #     best_auc_exp['test_acc'] = test_acc
                #     best_auc_exp['test_prec'] = test_prec
                #     best_auc_exp['test_recall'] = test_recall
                #     best_models.append(best_auc_exp)
                #     print("gene: ", gene, "model: ", model_type, ", cancer_type: ", cancer_type,
                #           ", layers: ", num_layer, ", channels: ", channels, ", batch_size: ", batch_size, " lr: ", lr,
                #           ", test auc:", test_auc, ", test acc: ", test_acc, ", precision: ", test_prec, ", recall: ",
                #           test_recall)

                for gene_selection in ["first_degree_neighbors"]:

                    ### ELR ###
                    model_type = "ELR"
                    best_auc = 0
                    best_auc_model = None

                    # feature selection
                    if gene_selection == "first_degree_neighbors":
                        gene_set = neighbors_funcoup
                    else:
                        gene_set = Parse.get_genes_selection_set(gene_selection)
                        gene_set = [n for n in gene_set if n in X_train_copy.columns.values]
                    X_train = X_train_copy.loc[:, gene_set].copy()
                    X_val = X_val_copy.loc[:, gene_set].copy()
                    X_test = X_test_copy.loc[:, gene_set].copy()

                    for lr in [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]:
                        for ratio in [0, 0.3, 0.5, 0.7, 1]:

                            experiment = {
                                "setup": setup,
                                "gene": gene,
                                "num_mutated": num_mutated,
                                "model": model_type,
                                "graph": "NA",
                                "gene_selection": gene_selection,
                                "cancer type": cancer_type,
                                "num_layer": "NA",
                                "channels": "NA",
                                "batch": "NA",
                                "lr": lr,
                                "ratio": ratio,
                                "dropout": "NA",
                                "aggregation": "NA",
                                "criterion": "NA",
                                "max_features": "NA",
                                "n_estimator": "NA",
                            }

                            model = ElasticNet_with_Gene_Vector(alpha=lr, l1_ratio=ratio, tol=0.01)
                            model.fit(X_train, y_train)
                            model.gene_vector = gene_set
                            y_val_hat = model.predict(X_val)
                            # calculate, print and save results
                            val_auc = sklearn.metrics.roc_auc_score(y_val, y_val_hat)
                            val_acc = sklearn.metrics.accuracy_score(y_val, list(map(int, y_val_hat > 0.5)))
                            val_prec = sklearn.metrics.precision_score(y_val, list(map(int, y_val_hat > 0.5)))
                            val_recall = sklearn.metrics.recall_score(y_val, list(map(int, y_val_hat > 0.5)))
                            print("gene/pathway: ", gene, "model: ", model_type, "cancer_type: ", cancer_type,
                                  ", lr: ", lr, ", ratio: ",
                                  ratio, ", VAL: auc:", val_auc, ", acc: ", val_acc, ", precision: ", val_prec,
                                  ", recall: ", val_recall)
                            experiment["val_auc"] = val_auc
                            experiment["val_acc"] = val_acc
                            experiment["val_prec"] = val_prec
                            experiment["val_recall"] = val_recall
                            results.append(experiment)
                            if val_auc > best_auc:
                                best_auc = val_auc
                                best_auc_model = model
                                best_auc_exp = experiment.copy()
                            del model
                    pred_df_auc = Parse.save_predictions_table(X_train, y_train, X_val,
                                                               y_val, X_test, y_test, best_auc_model, model_type)
                    pred_df_auc.to_csv(
                        PATH_save_results + "Prediction_tables/" + model_type + "/pred_table_Perturbations_RTK_RAS_label" + "_" + setup + "_" + model_type + "_" +
                        cancer_type + "_" + gene + "_" + gene_selection + "_AUC_" + str(best_auc) + ".csv")


                    filename = PATH_save_results + "Model_files/" + model_type + "/best_model_Perturbations_RTK_RAS_label" + "_" + setup + "_" + model_type + "_" + cancer_type + "_" + gene_selection + "_AUC_" + str(
                        best_auc) + ".sav"
                    pickle.dump(best_auc_model, open(filename, 'wb'))

                    ### run best model on Test ###
                    # run prediction on test set and produce scores
                    y_test_hat = best_auc_model.predict(X_test)

                    test_auc = sklearn.metrics.roc_auc_score(y_test, y_test_hat)
                    test_acc = sklearn.metrics.accuracy_score(y_test, list(map(int, y_test_hat > 0.5)))
                    test_prec = sklearn.metrics.precision_score(y_test, list(map(int, y_test_hat > 0.5)))
                    test_recall = sklearn.metrics.recall_score(y_test, list(map(int, y_test_hat > 0.5)))

                    best_auc_exp['test_auc'] = test_auc
                    best_auc_exp['test_acc'] = test_acc
                    best_auc_exp['test_prec'] = test_prec
                    best_auc_exp['test_recall'] = test_recall
                    best_models.append(best_auc_exp)

                    print("gene/pathway: ", gene, "model: ", model_type, ", cancer_type: ", cancer_type, ", ratio: ",
                          ratio, " lr: ", lr,
                          ", test auc:", test_auc, ", test acc: ", test_acc, ", precision: ", test_prec, ", recall: ",
                          test_recall)

                    # RF
                    model_type = "RF"
                    best_auc = 0
                    best_auc_model = None

                    for n_estimator in [1000, 1500, 2000]:
                        for criterion in ["gini", "entropy"]:
                            for max_features in [0.4, 0.3, 0.2, 0.1, "sqrt"]:
                                experiment = {
                                    "setup": setup,
                                    "gene": gene,
                                    "model": model_type,
                                    "graph": "NA",
                                    "gene_selection": "NA",
                                    "cancer type": cancer_type,
                                    "num_layer": "NA",
                                    "channels": "NA",
                                    "batch": "NA",
                                    "lr": "NA",
                                    "ratio": "NA",
                                    "dropout": dropout,
                                    "aggregation": "NA",
                                    "criterion": criterion,
                                    "max_features": max_features,
                                    "n_estimator": n_estimator,
                                }

                                model = RandomForest_with_Gene_Vector(n_estimators=n_estimator, max_depth=3,
                                                               criterion=criterion, max_features=max_features)
                                model.fit(X_train, y_train)
                                model.gene_vector = gene_set
                                y_val_hat = model.predict(X_val)
                                # calculate, print and save results
                                val_auc = sklearn.metrics.roc_auc_score(y_val, y_val_hat)
                                val_acc = sklearn.metrics.accuracy_score(y_val, list(map(int, y_val_hat > 0.5)))
                                val_prec = sklearn.metrics.precision_score(y_val, list(map(int, y_val_hat > 0.5)))
                                val_recall = sklearn.metrics.recall_score(y_val, list(map(int, y_val_hat > 0.5)))
                                print("model: ", model_type, "cancer_type: ", cancer_type, ", gene selection: ",
                                      gene_selection,
                                      ", criterion: ", criterion, ", max_features: ", max_features, ", n_estimator: ",
                                      n_estimator,
                                      ", auc:", val_auc, ", acc: ", val_acc, ", precision: ", val_prec, ", recall: ",
                                      val_recall)
                                experiment["val_auc"] = val_auc
                                experiment["val_acc"] = val_acc
                                experiment["val_prec"] = val_prec
                                experiment["val_recall"] = val_recall
                                results.append(experiment)
                                if val_auc > best_auc:
                                    best_auc = val_auc
                                    best_auc_model = model
                                    best_auc_exp = experiment.copy()
                                del model
                    pred_df_auc = Parse.save_predictions_table(X_train, y_train, X_val, y_val, X_test,
                                                               y_test, best_auc_model, model_type)
                    pred_df_auc.to_csv(
                        PATH_save_results + "Prediction_tables/" + model_type + "/pred_table_Perturbations_RTK_RAS_label" + "_" + setup + "_" + model_type + "_" +
                        cancer_type + "_" + gene + "_" + gene_selection + "_AUC_" + str(best_auc) + ".csv")
                    filename = (PATH_save_results + "Model_files/" + model_type + "/best_model_Perturbations_RTK_RAS_label" + "_" + setup + "_" + model_type + "_" +
                                cancer_type + "_" + gene + "_" + gene_selection + "_AUC_" + str(
                        best_auc) + ".sav")
                    pickle.dump(best_auc_model, open(filename, 'wb'))

                    ### run best model on Test ###
                    # run prediction on test set and produce scores
                    y_test_hat = best_auc_model.predict_proba(X_test)[:, 1]

                    test_auc = sklearn.metrics.roc_auc_score(y_test, y_test_hat)
                    test_acc = sklearn.metrics.accuracy_score(y_test, list(map(int, y_test_hat > 0.5)))
                    test_prec = sklearn.metrics.precision_score(y_test, list(map(int, y_test_hat > 0.5)))
                    test_recall = sklearn.metrics.recall_score(y_test, list(map(int, y_test_hat > 0.5)))

                    best_auc_exp['test_auc'] = test_auc
                    best_auc_exp['test_acc'] = test_acc
                    best_auc_exp['test_prec'] = test_prec
                    best_auc_exp['test_recall'] = test_recall

                    best_models.append(best_auc_exp)

            # Save results
            keys = results[0].keys()
            with open(
                    PATH_save_results + "Comparison_grid/grid_Perturbations_analysis_3_ELR_RF.csv",
                    'w', newline='') as output_file:
                dict_writer = csv.DictWriter(output_file, keys)
                dict_writer.writeheader()
                dict_writer.writerows(results)

            keys = best_models[0].keys()
            with open(
                    PATH_save_results + "Comparison_grid/best_models_Perturbations_analysis_3_ELR_RF.csv",
                    'w', newline='') as output_file:
                dict_writer = csv.DictWriter(output_file, keys)
                dict_writer.writeheader()
                dict_writer.writerows(best_models)

keys = results[0].keys()
with open(
        PATH_save_results + "Comparison_grid/grid_Perturbations_analysis_3_ELR_RF.csv",
        'w', newline='') as output_file:
    dict_writer = csv.DictWriter(output_file, keys)
    dict_writer.writeheader()
    dict_writer.writerows(results)

keys = best_models[0].keys()
with open(
        PATH_save_results + "Comparison_grid/best_models_Perturbations_analysis_3_ELR_RF.csv",
        'w', newline='') as output_file:
    dict_writer = csv.DictWriter(output_file, keys)
    dict_writer.writeheader()
    dict_writer.writerows(best_models)


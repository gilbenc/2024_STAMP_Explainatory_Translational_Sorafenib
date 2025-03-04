

library(ggplot2)
# library(ggpubr)
library("RColorBrewer")
library(ROCR)
library(tidyr)
library(pROC)



### Load Data
# PATH for DGX
# PATH <- "~//Desktop//Gil//2024_STAMP_Explainatory_Translational_Sorafenib//"
# PATH for Lenovo
PATH <- "C://Users//gil_ben_cohen//Desktop//gil//research//2024_STAMP_Explainatory_Translational_Sorafenib//"
PATH_perturbations_grid <- paste0(PATH, "Data_generated//Perturbation_models//Comparison_grid//")

# Enlight predictions
PATH_ENLIGHT_prediction_files = "Data_generated//Perturbation_models//"
paste0(PATH, "Data_generated//Perturbation_models//STAMP_predictions//ENLIGHT//")
ENLIGHT_treatment_response <- read.csv(paste0(PATH, "Data_from_source//ENLIGHT//Enlight_drug_response_classification.csv"), row.names = 1)
ENLIGHT_treatment_response_LIHC <- ENLIGHT_treatment_response[ENLIGHT_treatment_response$Dataset == "Sorafenib",]
ENLIGHT_treatment_response_BRCA <- ENLIGHT_treatment_response[ENLIGHT_treatment_response$Dataset == "Sorafenib_2",]


### Functions ###
calculate_AUC <- function(model_ENLIGHT_preds){
  # pred <- prediction(model_ENLIGHT_preds$GCN_linear, model_ENLIGHT_preds$ENLIGHT_response)
  # perf <- performance(pred, "tpr", "fpr")
  roc_obj <- roc(model_ENLIGHT_preds$ENLIGHT_response, model_ENLIGHT_preds[,2])
  return(as.numeric(auc(roc_obj)))
}

### MAIN ###

# model files
model_files = list.files(PATH_ENLIGHT_prediction_files, pattern = "*.csv", full.names = TRUE)

results_files <- list.files(PATH_perturbations_grid, pattern = "*.csv", full.names = TRUE)
results_files_best_models <- results_files[grepl("best_models", results_files)]
results_files_grid <- results_files[grepl("grid_", results_files)]


## Copy rest of file and complete it to fit the grid files.
## I need to concatenate the csv files into one big table, than present results similarly to the ENLIGHT AUC analysis
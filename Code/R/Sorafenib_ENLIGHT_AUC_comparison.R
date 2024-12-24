### ROC Script for Natan ###

library(ggplot2)
# library(ggpubr)
library("RColorBrewer")
library(ROCR)
library(tidyr)
library(pROC)



### Load Data
PATH <- "~//Desktop//Gil//2024_STAMP_Explainatory_Translational_Sorafenib//"
# Enlight predictions
PATH_ENLIGHT_prediction_files = 
  "~//Desktop//Gil//2024_STAMP_Explainatory_Translational_Sorafenib//Data_generated//STAMP_predictions//ENLIGHT//"
ENLIGHT_treatment_response <- read.csv(paste0(PATH, "Data_from_source//ENLIGHT//Enlight_drug_response_classification.csv"), row.names = 1)
ENLIGHT_treatment_response_LIHC <- ENLIGHT_treatment_response[ENLIGHT_treatment_response$Dataset == "Sorafenib",]
ENLIGHT_treatment_response_BRCA <- ENLIGHT_treatment_response[ENLIGHT_treatment_response$Dataset == "Sorafenib_2",]

# model files
model_files = list.files(PATH_ENLIGHT_prediction_files, pattern = "*.csv", full.names = TRUE)

# Gene groups
# missing: VEGFR 1, 2, 3, PDGFR
Sorafenib_targets = c("BRAF", "FGFR1", "FLT3", "KIT", "RAF1", "RET")
RTK_RAS_sig = c("EGFR", "ERRFI1", "KRAS", "MET", "NF1", "RASA1")

# Vectors to define the models
# gene_group: 0 = sorafenib targets, 1 = RTK RAS dominant, 2 = RTK RAS pathway model
gene_group <- c()
# model_group: 0 if not pan cancer. 1 if pan cancer predicting for LIHC, 2 if pan cancer predicting for BRCA
model_group <- c()
# tumor type
tumor_type_group <- c()
for(file in model_files){
  
  file_name = basename(file)
  if(strsplit(file_name, "_")[[1]][5] %in% Sorafenib_targets){
    gene_group <- c(gene_group, 0)
  }
  if(strsplit(file_name, "_")[[1]][5] %in% RTK_RAS_sig){
    gene_group <- c(gene_group, 1)
  }
  if(strsplit(file_name, "_")[[1]][5] == "RTK RAS"){
    gene_group <- c(gene_group, 2)
  }
  tumor_type <- strsplit(file_name, "_")[[1]][6]
  tumor_type_group <- c(tumor_type_group, tumor_type)
  if(tumor_type == "LIHC" || (tumor_type == "pan" && grepl("_1_", file_name)))
    model_group <- c(model_group, 1)
  if(tumor_type == "BRCA" || (tumor_type == "pan" && grepl("_2_", file_name)))
    model_group <- c(model_group, 2)
}

count = 1
for(file in model_files){
  read.csv(file, row.names = 1)
  if()
  merge()
}


## A. create performance for ROC curve calculations
# label should be metabric's label for mutated/non-mutated TP53, pred_score is the model's prediction score
ELR_pred <- prediction(ELR_pred_score, label)
ELR_perf <- performance(pred, "tpr", "fpr")

GCN_pred <- prediction(GCN_pred_score, label)
GCN_perf <- performance(pred, "tpr", "fpr")

RF_pred <- prediction(RF_pred_score, label)
RF_perf <- performance(pred, "tpr", "fpr")


## B. create figure
tiff(paste0(PATH_plots, "METABRIC_ROC_3_Models_qualitative_3.0.tiff"),
     units="in", width=8, height=8, res=300)
par(cex.axis=2, cex.lab = 2, mar=c(5,6,4,1))
print(plot(get(paste0("perf_", tumor_type, "_GCN")), 
           col = brw_dark2[1], lwd = 7)) 
print(plot(get(paste0("perf_", tumor_type, "_ELR_mutsigdb")), add = TRUE,
           col = brw_dark2[2], lwd = 7))
print(plot(get(paste0("perf_", tumor_type, "_RF_deseq")), add = TRUE,
           col = brw_dark2[3], lwd = 7))
GCN_legend <- paste0("GCN (AUC = ", round(100*best_models$test_auc[best_models$model == "GCN" &
                                                                     best_models$cancer.type == tumor_type],2), "%)")
ELR_msig_legend <- paste0("ELR MutSigDB (AUC = ", round(100*best_models$test_auc[best_models$model == "ELR_mutsigdb" &
                                                                                   best_models$cancer.type == tumor_type],2), "%)")
ELR_deseq_legend <- paste0("ELR Deseq (AUC = ", round(100*best_models$test_auc[best_models$model == "ELR_deseq" &
                                                                                 best_models$cancer.type == tumor_type],2), "%)")
RF_msig_legend <- paste0("RF MutSigDB (AUC = ", round(100*best_models$test_auc[best_models$model == "RF_mutsigdb" &
                                                                                 best_models$cancer.type == tumor_type],2), "%)")
RF_deseq_legend <- paste0("RF Deseq (AUC = ", round(100*best_models$test_auc[best_models$model == "RF_deseq" &
                                                                               best_models$cancer.type == tumor_type],2), "%)")

legend("bottomright",
       legend=c(GCN_legend, ELR_msig_legend, RF_deseq_legend),
       col=brw_dark2, 
       lwd=7, cex =1.5, xpd = TRUE, horiz = F)



dev.off()
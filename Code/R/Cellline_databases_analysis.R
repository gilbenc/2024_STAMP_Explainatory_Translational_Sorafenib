

library(ggplot2)
# library(ggpubr)
library("RColorBrewer")
library(ROCR)
library(tidyr)
library(pROC)



### Load Data
# PATH for DGX
PATH <- "~//Desktop//Gil//2024_STAMP_Explainatory_Translational_Sorafenib//"
# PATH for Lenovo
# PATH <- "C://Users//gil_ben_cohen//Desktop//gil//research//2024_STAMP_Explainatory_Translational_Sorafenib//"

PATH_data_source <- paste0(PATH, "Data_from_source//")
PATH_GDSC <- paste0(PATH_data_source, "GDSC//")
PATH_CCLE <- paste0(PATH_data_source, "CCLE//")

CCLE_cell_lines_data <- read.csv(paste0(CCLE_PATH, "Cell_lines_annotations_20181226.txt"))
CCLE_respurposing_PRISM <- read.csv(paste0(CCLE_PATH, "284461-73-0 (BRDBRD-K23984367-075-15-2) PRISM Repurposing Public 24Q2.csv"))
CCLE_drug_response_CTD <- read.csv(paste0(CCLE_PATH, "284461-73-0 (CTRP349006) Drug sensitivity AUC (CTD^2).csv"))
CCLE_drug_response_GDSC1 <- read.csv(paste0(CCLE_PATH, "284461-73-0 (GDSC130) Drug sensitivity IC50 (Sanger GDSC1).csv"))
CCLE_drug_response_PRISM <- read.csv(paste0(CCLE_PATH, "284461-73-0 (BRDBRD-K23984367-001-07-5) Drug sensitivity AUC (PRISM Repurposing Secondary Screen).csv"))

GDSC_cell_lines_data <- read.csv(paste0(GDSC_PATH, "GDSC_cell_lines_drug_combinations_logIC50_TableS4A.xlsx"))
CCLE_GDSC_intersection <- read.csv(paste0(PATH_GDSC, "GDSC_CCLE_overlapping_cell_lines_TableS4E.xlsx"))



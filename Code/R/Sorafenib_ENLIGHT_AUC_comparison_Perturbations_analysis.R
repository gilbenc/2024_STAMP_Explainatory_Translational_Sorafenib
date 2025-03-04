

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


# Gene groups
# missing: VEGFR 1, 2, 3, PDGFR
Sorafenib_targets = c("BRAF", "FGFR1", "FLT3", "KIT", "RAF1", "RET")
RTK_RAS_sig = c("EGFR", "ERRFI1", "KRAS", "MET", "NF1", "RASA1")


# Initialize an empty data frame to store results
results <- data.frame(
  model = character(),
  gene = character(),
  gene_inclusion = character(),
  tumor_type = character(),
  data_type = character(),
  auc = numeric(),
  stringsAsFactors = FALSE
)

for(file in model_files){
  file_name = basename(file)
  print(file_name)
  all = 0
  
  gene_group <- strsplit(file_name, "_")[[1]][4]
  if(gene_group == "sorafenib"){
    gene_group <- "Sorafenib_targets"
  }
  if(gene_group == "rtk"){
    gene_group <- "RTK_RAS_dominant_genes"
    all = all + 1
  }
  
  
  gene <- strsplit(file_name, "_")[[1]][7+all]
  if(gene == "all"){
    gene = "All neighbors"
    all = all+1
  }
  
  
  
  tumor <- strsplit(file_name, "_")[[1]][8+all]
  if(tumor == "pan") {
    tumor <- "pan_cancer"
    all = all + 1
    model <- strsplit(file_name, "_")[[1]][9+all]
  } else {
    model <- strsplit(file_name, "_")[[1]][9+all]
  }
  
  if(tumor == "LIHC" || (tumor == "pan_cancer" && grepl("_1_", file_name)))
    data <- "LIHC"
  if(tumor == "BRCA" || (tumor == "pan_cancer" && grepl("_2_", file_name)))
    data <- "BRCA"
  
  print(paste0("model: ", model, ", gene: ", gene, ", gene group:", gene_group, ", tumor type:", tumor, ", data: ", data))
  model_ENLIGHT_preds <- read.csv(file, row.names = 1)
  if(data == "LIHC"){
    model_ENLIGHT_preds$ENLIGHT_response <- ENLIGHT_treatment_response_LIHC[row.names(model_ENLIGHT_preds), "Response"]} else {
      model_ENLIGHT_preds$ENLIGHT_response <- ENLIGHT_treatment_response_BRCA[row.names(model_ENLIGHT_preds), "Response"] }
  if(is.numeric(model_ENLIGHT_preds[,2])){
    auc <- calculate_AUC(model_ENLIGHT_preds)  
  } else {
    print("failed for this model")
    next
  }
  
  
  # Add the result to the data frame
  results <- rbind(
    results,
    data.frame(
      model = model,
      gene = gene,
      gene_inclusion = gene_group,
      tumor_type = tumor,
      data_type = data,
      auc = auc,
      stringsAsFactors = FALSE
    ))  
}


# Order tumor types and gene inclusion to control plot grouping
results$tumor_type <- factor(results$tumor_type, levels = c("pan_cancer", "LIHC", "BRCA"))
results$gene_inclusion[results$gene == "All neighbors"] <- "All neighbors"
results$gene_inclusion <- factor(results$gene_inclusion, 
                                 levels = c("All neighbors","Sorafenib_targets", "RTK_RAS_dominant_genes"))
results$model <- factor(results$model, levels = c("GCN", "ELR", "RF"))

# Filter data for LIHC
results_LIHC <- subset(results, data_type == "LIHC")
results_LIHC_ELR <- subset(results_LIHC, model == "ELR")
results_LIHC_RF <- subset(results_LIHC, model == "RF")
results_LIHC_GCN <- subset(results_LIHC, model == "GCN")
# Filter data for BRCA
results_BRCA <- subset(results, data_type == "BRCA")
results_BRCA_ELR <- subset(results_BRCA, model == "ELR")
results_BRCA_RF <- subset(results_BRCA, model == "RF")
results_BRCA_GCN <- subset(results_BRCA, model == "GCN")


# Heatmap for LIHC
heatmap_LIHC_GCN <- ggplot(results_LIHC_GCN, aes(x = tumor_type, y = gene, fill = auc)) +
  geom_tile(color = "white") +
  scale_fill_gradient2(
    low = "blue", mid = "yellow", high = "red", midpoint = 0.67, limits = c(0.41, 0.925), # Set the range of the color scale
    name = "AUC",
    breaks = c(0.41, 0.67, 0.925)
  ) +
  facet_grid(rows = vars(gene_inclusion), scales = "free_y", space = "free_y") +
  labs(
    title = "AUC: GCN LIHC Models",
    x = "Tumor Type",
    y = "Gene"
  ) +
  theme_minimal(base_size = 32) +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    strip.text.y = element_text(angle = 0, face = "bold"),
    plot.title = element_text(hjust = 0.5)
  )


heatmap_LIHC_ELR <- ggplot(results_LIHC_ELR, aes(x = tumor_type, y = gene, fill = auc)) +
  geom_tile(color = "white") +
  scale_fill_gradient2(
    low = "blue", mid = "yellow", high = "red", midpoint = 0.63, limits = c(0.41, 0.925), # Set the range of the color scale
    name = "AUC",
    breaks = c(0.41, 0.63, 0.925)
  ) +
  facet_grid(rows = vars(gene_inclusion), scales = "free_y", space = "free_y") +
  labs(
    title = "AUC: ELR LIHC Models",
    x = "Tumor Type",
    y = "Gene"
  ) +
  theme_minimal(base_size = 32) +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    strip.text.y = element_text(angle = 0, face = "bold"),
    plot.title = element_text(hjust = 0.5)
  )

heatmap_LIHC_RF <- ggplot(results_LIHC_RF, aes(x = tumor_type, y = gene, fill = auc)) +
  geom_tile(color = "white") +
  scale_fill_gradient2(
    low = "blue", mid = "yellow", high = "red", midpoint = 0.67, limits = c(0.41, 0.925), # Set the range of the color scale
    name = "AUC",
    breaks = c(0.41, 0.67, 0.925)
  ) +
  facet_grid(rows = vars(gene_inclusion), scales = "free_y", space = "free_y") +
  labs(
    title = "AUC: RF LIHC Models",
    x = "Tumor Type",
    y = "Gene"
  )+
  theme_minimal(base_size = 32) +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    strip.text.y = element_text(angle = 0, face = "bold"),
    plot.title = element_text(hjust = 0.5)
  )


## heatmaps for BRCA
heatmap_BRCA_GCN <- ggplot(results_BRCA_GCN, aes(x = tumor_type, y = gene, fill = auc)) +
  geom_tile(color = "white") +
  scale_fill_gradient2(
    low = "blue", mid = "yellow", high = "red", midpoint = 0.59, limits = c(0.41, 0.761), # Set the range of the color scale
    name = "AUC",
    breaks = c(0.41, 0.59, 0.76)
  ) +
  facet_grid(rows = vars(gene_inclusion), scales = "free_y", space = "free_y") +
  labs(
    title = "AUC: GCN Breast Models",
    x = "Tumor Type",
    y = "Gene"
  )+
  theme_minimal(base_size = 32) +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    strip.text.y = element_text(angle = 0, face = "bold"),
    plot.title = element_text(hjust = 0.5)
  )

heatmap_BRCA_ELR <- ggplot(results_BRCA_ELR, aes(x = tumor_type, y = gene, fill = auc)) +
  geom_tile(color = "white") +
  scale_fill_gradient2(
    low = "blue", mid = "yellow", high = "red", midpoint = 0.59, limits = c(0.41, 0.761), # Set the range of the color scale
    name = "AUC",
    breaks = c(0.41, 0.59, 0.76)
  ) +
  facet_grid(rows = vars(gene_inclusion), scales = "free_y", space = "free_y") +
  labs(
    title = "AUC: ELR Breast Models",
    x = "Tumor Type",
    y = "Gene"
  ) +
  theme_minimal(base_size = 32) +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    strip.text.y = element_text(angle = 0, face = "bold"),
    plot.title = element_text(hjust = 0.5)
  )

heatmap_BRCA_RF <- ggplot(results_BRCA_RF, aes(x = tumor_type, y = gene, fill = auc)) +
  geom_tile(color = "white") +
  scale_fill_gradient2(
    low = "blue", mid = "yellow", high = "red", midpoint = 0.59, limits = c(0.41, 0.761), # Set the range of the color scale
    name = "AUC",
    breaks = c(0.41, 0.59, 0.76)
  ) +
  facet_grid(rows = vars(gene_inclusion), scales = "free_y", space = "free_y") +
  labs(
    title = "AUC: RF Breast Models",
    x = "Tumor Type",
    y = "Gene"
  ) +
  theme_minimal(base_size = 32) +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    strip.text.y = element_text(angle = 0, face = "bold"),
    plot.title = element_text(hjust = 0.5)
  )
print(heatmap_LIHC_GCN)
print(heatmap_LIHC_ELR)
print(heatmap_LIHC_RF)
print(heatmap_BRCA_GCN)
print(heatmap_BRCA_ELR)
print(heatmap_BRCA_RF)

### Test for significance of results' properties in  ENLIGHT LIHC:
kruskal_LIHC_model <- kruskal.test(auc ~ model, data = results_LIHC)
kruskal_LIHC_gene <- kruskal.test(auc ~ gene, data = results_LIHC)
# kruskal_LIHC_gene_pan <- kruskal.test(auc ~ gene, data = results_LIHC[results_LIHC$tumor_type == "pan_cancer",])
kruskal_LIHC_gene_inclusion <- kruskal.test(auc ~ gene_inclusion, data = results_LIHC)
# kruskal_LIHC_gene_inclusion_pan <- kruskal.test(auc ~ gene_inclusion, data = results_LIHC[results_LIHC$tumor_type == "pan_cancer",])
kruskal_LIHC_tumor_type <- kruskal.test(auc ~ tumor_type, data = results_LIHC)
# kruskal_LIHC_tumor_type_dominant <- kruskal.test(auc ~ tumor_type, data = results_LIHC[results_LIHC$gene_inclusion == "RTK_RAS_dominant_genes",])
print(kruskal_LIHC_model)
print(kruskal_LIHC_gene)
# print(kruskal_LIHC_gene_pan)
print(kruskal_LIHC_gene_inclusion)
print(kruskal_LIHC_tumor_type)
# print(kruskal_LIHC_gene_inclusion_pan)
# print(kruskal_LIHC_tumor_type_dominant)

### Test for significance of results' properties in  ENLIGHT BRCA:
kruskal_BRCA_model <- kruskal.test(auc ~ model, data = results_BRCA)
kruskal_BRCA_gene <- kruskal.test(auc ~ gene, data = results_BRCA)
kruskal_BRCA_gene_inclusion <- kruskal.test(auc ~ gene_inclusion, data = results_BRCA)
# kruskal_BRCA_gene_pan <- kruskal.test(auc ~ gene, data = results_BRCA[results_BRCA$tumor_type == "pan_cancer",])
# kruskal_BRCA_gene_inclusion_pan <- kruskal.test(auc ~ gene_inclusion, data = results_BRCA[results_BRCA$tumor_type == "pan_cancer",])
kruskal_BRCA_tumor_type <- kruskal.test(auc ~ tumor_type, data = results_BRCA)

print(kruskal_BRCA_model)
print(kruskal_BRCA_gene)
print(kruskal_BRCA_gene_inclusion)
# print(kruskal_BRCA_gene_pan)
# print(kruskal_BRCA_gene_inclusion_pan)

print(kruskal_BRCA_tumor_type)




# Reorder genes based on mean AUC within each gene_inclusion group
library(dplyr)
library(ggplot2)


# Calculate mean AUC for each gene within each gene_inclusion category
mean_auc_order <- results_LIHC %>%
  group_by(gene_inclusion, gene) %>%
  summarize(mean_auc = mean(auc, na.rm = TRUE), .groups = "drop") %>%
  arrange(gene_inclusion, desc(mean_auc))
# very specific code- DONT USE IF NEW DATA.


results_LIHC_Sorafenib <- results_LIHC[results_LIHC$gene_inclusion == "Sorafenib_targets",]

# Reorder genes based on the calculated order
ggplot(results_LIHC, aes(
  x = factor(gene, levels = mean_auc_order$gene),
  y = auc, fill = gene_inclusion
)) +
  geom_boxplot() +
  labs(
    title = "Perturbations ENLIGHT LIHC validation: AUC Distribution by Gene groups",
    x = "Gene", y = "AUC", fill = "Gene Group"
  ) +
  theme_minimal(base_size = 26) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) # Rotate x-axis labels for better readability



ggplot(results_LIHC, aes(
  x = gene_inclusion,
  y = auc, fill = model
)) +
  geom_boxplot() +
  labs(
    title = "Perturbations ENLIGHT LIHC validation: AUC Distribution by Models",
    x = "Gene Group", y = "AUC", fill = "Model"
  ) +
  theme_minimal(base_size = 26) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) # Rotate x-axis labels for better readability


## ENLIHT BRCA setup:
## Plot based on tumor type training:
# Reorder genes based on the calculated order
ggplot(results_BRCA, aes(
  x = factor(tumor_type),
  y = auc, fill = tumor_type
)) +
  geom_boxplot() +
  labs(
    title = "ENLIGHT BRCA validation: AUC Distribution training setup",
    x = "Tumor type", y = "AUC"
  ) +
  theme_minimal(base_size = 26) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) # Rotate x-axis labels for better readability

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

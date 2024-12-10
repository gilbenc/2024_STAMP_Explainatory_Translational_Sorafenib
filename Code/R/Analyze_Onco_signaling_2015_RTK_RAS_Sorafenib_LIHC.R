#######################################################################################################
###                                                                                                 ###
###       Title: Analyze Oncogenic signaling data                                                   ###
###       Objective: Decide how to further use this data for Sorafenib project (train new models?)  ###
###                                                                                                 ###
#######################################################################################################

# Variables
Sorafenib_targets <- c("RAF1", "BRAF", "VEGFR1", "VEGFR2", "VEGFR3", "PDGFR", "KIT", "FLT3", "FGFR1", "RET")

# Functions
get_sorafenib_targets_per_tumor_type <- function(tumor_type){
  TCGA_data <- TCGA_data_from_cbioportal[TCGA_data_from_cbioportal$TCGA.PanCanAtlas.Cancer.Type.Acronym == tumor_type,]
  onco_map_data <- Oncogenic_signaling_map_genes[Oncogenic_signaling_map_genes$SAMPLE_BARCODE %in% TCGA_data$Sample.ID,]
  onco_map_data <- onco_map_data[,colnames(onco_map_data) %in% Sorafenib_targets]
  print(tumor_type)
  print(apply(onco_map_data, 2, sum))
}

# Data
Oncogenic_signaling_map_genes <- read.csv("Desktop//Gil//2024_Sorafenib_Explainatory_STAMP//Data_from_source//Oncogenic_signaling_pathways//Oncogenic_signaling_pathways_cell_paper_table_S4_genomic_alteration_matrices1_genes.csv", sep =  ",")
row.names(Oncogenic_signaling_map_genes) <- Oncogenic_signaling_map_genes$SAMPLE_BARCODE
Oncogenic_signaling_map_pathways <- read.csv("Desktop//Gil//2024_Sorafenib_Explainatory_STAMP//Data_from_source//Oncogenic_signaling_pathways//Oncogenic_signaling_pathways_cell_paper_table_S4_genomic_alteration_matrices1_pathways.csv", sep =  ",")
row.names(Oncogenic_signaling_map_pathways) <- Oncogenic_signaling_map_pathways$SAMPLE_BARCODE
TCGA_data_from_cbioportal <-  read.csv("Desktop//Gil//2024_Sorafenib_Explainatory_STAMP//Data_from_source//Cbioportal//Cbioportal_Pan_Cacner_WmutProfile_clinical_data.tsv", sep =  "\t")
Oncogenic_signaling_map_genes_to_pathways_RTK_RAS <- read.csv("Desktop//Gil//2024_Sorafenib_Explainatory_STAMP//Data_from_source//Oncogenic_signaling_pathways//Oncogenic_signaling_pathways_cell_paper_table_S3_pathways_to_gene_alteration_RTK_RAS.csv", sep = ",")
table(TCGA_data_from_cbioportal$TCGA.PanCanAtlas.Cancer.Type.Acronym)

# get TCGA HCC samples
TCGA_cbioportal_LIHC_samples <- TCGA_data_from_cbioportal[TCGA_data_from_cbioportal$TCGA.PanCanAtlas.Cancer.Type.Acronym == "LIHC",]

# get onco maps HCC samples (genes & pathways)
Onco_map_genes_LIHC <- Oncogenic_signaling_map_genes[Oncogenic_signaling_map_genes$SAMPLE_BARCODE %in% TCGA_cbioportal_LIHC_samples$Sample.ID,]
Onco_map_pathways_LIHC <- Oncogenic_signaling_map_pathways[Oncogenic_signaling_map_pathways$SAMPLE_BARCODE %in% TCGA_cbioportal_LIHC_samples$Sample.ID,]
# only sorafenib targets
Onco_map_sorafenib_target_genes_LIHC <- Onco_map_genes_LIHC[,colnames(Onco_map_genes_LIHC) %in% Sorafenib_targets]
# only genes in RTK RAS pathway
Onco_map_RTK_RAS_genes_LIHC <- Onco_map_genes_LIHC[,colnames(Onco_map_genes_LIHC) %in% Oncogenic_signaling_map_genes_to_pathways_RTK_RAS$Gene]

apply(Onco_map_RTK_RAS_genes_LIHC, 2, sum)

sapply(unique(TCGA_data_from_cbioportal$TCGA.PanCanAtlas.Cancer.Type.Acronym), get_sorafenib_targets_per_tumor_type)

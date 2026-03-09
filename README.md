# WMH-evolution-phenotypes
This repository contains the analysis scripts and derived results supporting the study “In vivo identification of distinct phenotypes in white matter lesion evolution”.
The file WMH_clusters_metrics-all_with_clinic_final.xlsx contains all the information used in the analyses. This dataset includes multimodal metrics for each identified cluster, along with demographic and clinical information at the subject level.
The script clustering_final should be executed using this Excel file as input. This script performs the cluster analyses and the associated statistical tests.
Once this step is completed, the script susceptibility_final should be run. This script uses the outputs generated in the previous step to perform the susceptibility association analyses.

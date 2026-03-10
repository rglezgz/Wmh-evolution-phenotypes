WMH-evolution-phenotypes

This repository contains the analysis scripts and derived results supporting the study:
“In vivo identification of distinct phenotypes in white matter lesion evolution.”

The file WMH_clusters_metrics-all_with_clinic_final.xlsx contains all the data used in the analyses. This dataset includes multimodal metrics for each identified cluster, along with demographic and clinical information at the subject level.

Analysis workflow

1- Cluster analysis

Run the script clustering_final using the Excel file as input.
This script performs the clustering analyses and the associated statistical tests.

2- Susceptibility association analysis

After completing the clustering step, run the script susceptibility_final.
This script uses the outputs generated in the previous step to perform the susceptibility association analyses.

Results

The folder Clustering contains all the results generated after running both scripts.

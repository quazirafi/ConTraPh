# ConTraPh

This repository contains the code and dataset of the ICS 2025 paper "ConTraPh: Contrastive Learning for Parallelization and Performance Optimization".

# Reduction Style Detection

1. Download `dgl-csv-reduction-styles-better.zip'  and `model-reduction-style-best-model-class-better-30.pt' from https://zenodo.org/records/11003882.
2. Unzip `dgl-csv-reduction-styles-better.zip' and move it to the ./Datamodel directory and unzip the file.
3. Move `model-reduction-style-best-model-class-better-30.pt' to ./ModelChekpoints directory.
4. Run contrastive-learning-red-styles.py

# Schedule Style Detection

1. Download `dgl-csv-scheduling-style.zip'  and `model-scheduling-style-best-model-804-100.pt' from https://zenodo.org/records/11003951.
2. Unzip `dgl-csv-scheduling-style.zip' and move it to the ./Datamodel directory and unzip the file.
3. Move `model-scheduling-style-best-model-804-100.pt' to ./ModelChekpoints directory.
4. Run contrastive-learning-sched-styles.py

# Simd/Target Configuration Prediction

1. Go to ./Datamodel directory. Unzip `dgl-csv-simd-target-mod.zip'
2. Run contrastive-learning-simd-target.py





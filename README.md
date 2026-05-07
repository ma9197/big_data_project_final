# US Flight Delays: Graph-Augmented Delay Prediction

A big data project analyzing US domestic flight delays (2019–2023) using PySpark, graph analytics, and machine learning. The core hypothesis: **airport network structure (graph features) improves delay prediction** beyond operational signals alone.

## Overview

| Item | Detail |
|------|--------|
| Dataset | BTS On-Time Reporting, Jan 2019 to Dec 2023 |
| Flights analyzed | ~30.8 million completed flights |
| Graph | 381 airports (nodes), 8,128 directed routes (edges) |
| Task | Binary classification: will a flight arrive ≥15 min late? |
| Best AUC | ~0.64 (balanced Random Forest) |

## Repository Structure

```
CSVs/                          # Raw BTS monthly CSVs (stored via Git LFS)
  T_ONTIME_REPORTING_YYYY_MM.csv
Code/
  Step1_Data_Ingestion_Cleaning.ipynb
  Step2_Graph_Construction.ipynb
  Step3_Graph_Analytics.ipynb
  Step4_Modeling.ipynb
  Step4B_Balanced_Models.ipynb
```

## Pipeline

### Step 1: Data Ingestion & Cleaning
Load 60 monthly CSVs with PySpark, cast types, remove nulls, filter to completed flights. Outputs cleaned Parquet partitioned by year/month.

### Step 2: Graph Construction
Build a directed airport-route graph enriched with OpenFlights metadata.
- **Nodes**: 381 airports with traffic counts, delay rates, geographic coordinates
- **Edges**: 8,128 routes with avg departure/arrival delay, % delayed ≥15 min, distance

### Step 3: Graph Analytics
Compute centrality metrics with NetworkX to identify delay-critical hubs and fragile routes:
- Degree centrality, PageRank (traffic & delay-weighted), betweenness centrality
- Delay propagation score per airport
- Route fragility score (traffic × delay rate)

### Step 4: Modeling (Baseline)
Train Logistic Regression and Random Forest classifiers with PySpark MLlib.
- AUC ~0.64
- Severe class imbalance (82% on-time) → recall ≈ 0 on delayed class → addressed in Step 4B

### Step 4B: Balanced Models
Fix class imbalance with per-row class weights and optimal threshold tuning.
- Decision thresholds: LR 0.50, RF 0.52 (maximizes F1 on delayed class)
- Final delayed-class metrics: Precision ~0.28 / Recall ~0.58 / F1 ~0.39
- Network features (`origin_pagerank_traffic`, `betweenness`, etc.) rank **2nd-10th** in feature importance, validating the graph-augmentation approach

## Key Finding

Graph/network features consistently rank among the top predictors across both balanced models. This confirms that **where a flight originates in the airport network** is a meaningful signal for delay prediction, beyond just operational features like carrier, time of day, or distance.

## Requirements

```
pyspark
networkx
scikit-learn
pandas
numpy
matplotlib
```

## Data Source

Raw data downloaded from the [BTS TranStats On-Time Reporting database](https://www.transtats.bts.gov/Tables.asp?QO_VQ=EFD&QO_anzr=Nv4yv0r%FDb0-gv`r%FDcr4s14`n0pr%FD%FLS14%FDb0r-Fvzr%FDQn6n&QO_fu146_anzr=b0-gv`r). Files follow the naming convention `T_ONTIME_REPORTING_YYYY_MM.csv`.

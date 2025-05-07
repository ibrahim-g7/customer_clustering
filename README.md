# Customer Segmentation Analysis

A K-means clustering analysis of customer data to identify distinct market segments and develop targeted marketing strategies.

- [Source Code](./src/notenook.ipynb)
- [Report](./doc/report.pdf)
- [Installation Instruction](#running-instruction)


## Overview

This project analyzes customer data from a supermarket mall membership database to segment customers based on:
- Age
- Annual Income
- Education Level

## Data

The dataset contains 2000 customer records with the following features:
- ID: Customer identifier
- Sex: Male/Female
- Marital Status: Single/Non-single
- Age: 18-76 years
- Education: Other/High School/University/Graduate
- Income: Annual income (USD)
- Occupation: Unemployed/Skilled/Management
- Settlement Size: Small/Mid/Big city

## Methods

1. Data preprocessing
   - Log transformation of income
   - Feature standardization
   - Feature selection

2. Clustering
   - K-means++ initialization
   - Optimal cluster selection using WCSS and Silhouette scores
   - Final model with 5 clusters

## Results

Identified 5 distinct customer segments with specific demographic and financial characteristics, enabling targeted marketing strategies for each group.


# Running Instructions

1. Create a python cirtual environment: 

```python3 -m venv .venv ```

2. Activate the environment: 

- On Linux and MacOS:

``` source .venv/bin/activate ```

- On Windows

``` .venv\Scripts\activate ``` 

3. Install the requirments/dependencies: 

``` pip install -r requirements.txt```

4. Run the code.

5. Deactivate when finished 

```shell
deactivate
```


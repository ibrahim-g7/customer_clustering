# Customer Segmentation Analysis

A K-means clustering analysis of customer data to identify distinct market segments and develop targeted marketing strategies.

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

## Files

- `segmentation_analysis.qmd`: Main analysis notebook
- `data/segmentation_data.csv`: Raw dataset
- `img/`: Visualization outputs

## Requirements

- Python 3.8+
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

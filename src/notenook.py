import marimo

__generated_with = "0.13.6"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Assignment 3 
    Ibrahim Alghrabi -- 201724510 

    ### Dataset 
    1. Consist of 2000 observation of purchasing behavior.
    2. Data dictionary is as follow:

    | Variable        | Data Type   | Range        | Description                                                                   |
    | :---------------: | :-----------: | :------------: | :----------------------------------------------------------------------------- |
    | ID              | Numerical   | Integer      | Unique customer identifier.                                                 |
    | Sex             | Categorical | {0, 1}       | Biological sex: 0 = male, 1 = female.                                       |
    | Marital Status  | Categorical | {0, 1}       | Marital status: 0 = single, 1 = non-single.                                 |
    | Age             | Numerical   | Integer      | Customer's age in years (18-76).                                              |
    | Education       | Categorical | {0, 1, 2, 3} | Education level: 0 = other/unknown, 1 = high school, 2 = university, 3 = graduate school |
    | Income          | Numerical   | Real         | Annual income in USD (35832-309364).                                        |
    | Occupation      | Categorical | {0, 1, 2}    | Occupation category: 0 = unemployed/unskilled, 1 = skilled/official, 2 = management/self-employed |
    | Settlement Size | Categorical | {0, 1, 2}    | City size: 0 = small, 1 = mid-sized, 2 = big.                                 |

    ### Task
    1. To Apply clustering techniques to analyze the customers and uncover patterns in customers demographic and financial behvaior using K-means algorithm.
    2. Use centroid initialization to improve convergence.
    3. Determine the optimal number os cluster using both WCSS and Silhouette Coefficient.
    4. Train and predict cluster label for each customer and append it as a new column to the dataset
    5. Visualize the clustering results:
       - Annual Income (x-axis) vs Age (y-axis).
       - Annual Income (x-axis) vs Education Level (y-axis).

    ### Interpretation 
    1. Provide a detailed description of each identified customer clustering.
    2. Propose specific marketing strategies taiolred to each customer segment.

    ### My Approach 

    1. Preprocessing.
    2. Implement K-Mean++
    3. Use WCSS.
    4. Calculate Silhouette Scores.
    5. Determine the optimal K
    6. Final Model Training.
    7. Predict.
    8. Visualize 
    """
    )
    return


@app.cell
def _():
    import os
    import marimo as mo
    import pandas as pd 
    import numpy as np 
    import matplotlib.pyplot as plt 
    import seaborn as sns
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import silhouette_score
    plt.style.use('seaborn-v0_8')
    # plt.style.use('default')

    return KMeans, StandardScaler, mo, np, os, pd, plt, silhouette_score, sns


@app.cell
def _(os):
    # Manage directories 
    src_dir = os.path.dirname(os.path.abspath(__file__))
    repo_dir = os.path.join(src_dir, os.pardir)
    data_dir = os.path.join(repo_dir, "data")
    data_path = os.path.join(data_dir, "segmentation_data.csv")
    data_path

    return (data_path,)


@app.cell
def _(data_path, pd):
    # Load datasets
    df = pd.read_csv(data_path)
    df.info()
    return (df,)


@app.cell
def _(mo):
    mo.md(
        r"""
    # Dataset 

    1. The dataset contains 2000 obbservation and 8 numerical features.
    2. The data does not have any missing values.
   
    From the analsys beloew, we notice the following: 
    
    1. Male custmer are slightly higher than demalre customers. 
    2. Equaly distributed marital status. 
    3. The age distribution is right skewed, with mean of 35 years and standard deviation of ~12 years.
    4. Most of the customer are high school graduate. 
    5. Income distribution is also right skewed with an avergae of 120,000$ and stanadrad deviation of 38000$.
    6. Most customers are employed or self employed.
    7. Most customers live in small cities. 
    """
    )
    return


@app.cell
def _(df, plt, sns):
    cols = df.columns.to_list()[1:]
    fig, axes = plt.subplots(2, 4, figsize=(18,6))
    axes = axes.flatten()
    fig.suptitle("Bar plots of the dataset")
    for idx, col in enumerate(cols): 
        x = df[col]
        sns.histplot(x, ax=axes[idx])
        plt.tight_layout()
    plt.show()
    return


@app.cell
def _(df):
    df.describe()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    # Preprocessing 

    1. Pick only relevant features. Income, Age, and Education  
    2. Apply log transformation to income column since it shows a noticeable skewness 
    3. Scale the features using standard scaler. 
    """
    )
    return


@app.cell
def _(df):
    # 1. Drop id columns
    df.drop(columns="ID", axis=1, inplace=True)
    return


@app.cell
def _(df, np, plt, sns):
    _fig, _axes = plt.subplots(1, 2, figsize=(18,6))
    _fig.suptitle("Applying log transformation")
    sns.histplot(df["Income"], ax=_axes[0])
    _axes[0].set_title("Before")
    df["Income"] = np.log1p(df["Income"])
    sns.histplot(df["Income"], ax=_axes[1])
    _axes[1].set_title("After")
    plt.show()
    return


@app.cell
def _(StandardScaler, df):
    # Pick selected features 
    df_sel = df[["Income", "Age", "Education"]].copy()

    # Apply standard scaling
    scaler = StandardScaler()
    col_to_scale = ["Income", "Age"]
    df_sel[col_to_scale] = scaler.fit_transform(df_sel[col_to_scale])

    df_sel.describe()
    return col_to_scale, df_sel, scaler


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    #  K Value : 

    Determining K value with the following metrics while using KMeans++ initialize method. 

    1. WCSS.
    2. Silhouette Scores

    """
    )
    return


@app.cell
def _(KMeans, df_sel, np, silhouette_score):
    # Setting the ranges from 1 to 10

    k_range = np.arange(2, 16)
    wcss_scores = []
    sil_scores = []
    for k in k_range: 
        k_means= KMeans(
            n_clusters=k,
            init="k-means++" # KMeans++ init method
        )
        k_means.fit(df_sel)
        wcss = k_means.inertia_
        labels = k_means.labels_
        sil = silhouette_score(df_sel, labels)
        print(f"K:{k}. WCSS: {wcss:.4f}, Sillhouette: {sil:.4f}")
        wcss_scores.append(wcss)
        sil_scores.append(sil)

    return k_range, sil_scores, wcss_scores


@app.cell
def _(k_range, plt, sil_scores, sns, wcss_scores):
    _fig, _axes = plt.subplots(1, 2, figsize=(18,6))
    _axes[0].set_title("WCSS")
    _axes[0].set_xlabel("K")
    _axes[0].set_ylabel("WCSS")
    sns.lineplot(x=k_range, y=wcss_scores, ax=_axes[0])
    _axes[1].set_title("Silhouette Scores")
    _axes[1].set_xlabel("K")
    _axes[1].set_ylabel("Silhouette Scores")
    sns.lineplot(x=k_range, y=sil_scores, ax=_axes[1])

    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Picking K value. 

    From the above graphs we note: 

    1. Rate of dcreasing slows down (elbow) between $4 \leq  K \leq 5$.
    2. A highest score is around $K=5$. Note that different run could result in differnt graphs, but generaly the performance is better around $K=5$.

    Finally, considering the elbow point and interpretability and practical segmentation, we select $K=5$. 
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    # Final Model

    1. Train the final model with $K=5$.
    2. Predict the data.
    3. Append the dataframe
    4. Interperate the clusters.
    """
    )
    return


@app.cell
def _(KMeans, df_sel, silhouette_score):
    # Train the final model
    optimal_k = 5
    k_final = KMeans(
        n_clusters=optimal_k,
        init="k-means++"
    )
    # Predict the data
    k_final.fit(df_sel)
    wcss_final = k_final.inertia_
    labels_final = k_final.labels_
    sil_final = silhouette_score(df_sel, labels_final)

    # Append the the predicted labels
    df_sel["clusters"] = labels_final

    return


@app.cell
def _(col_to_scale, df_sel, np, scaler):
    # Reverse log transformation and scaling
    df_inverse = df_sel.copy()
    df_inverse[col_to_scale] = scaler.inverse_transform(df_inverse[col_to_scale])
    df_inverse["Income"] = np.expm1(df_inverse["Income"])
    df_inverse.describe()
    return (df_inverse,)


@app.cell
def _(df_inverse):
    _cols = ["Income", "Age", "Education"]
    print(df_inverse.groupby("clusters")["Income"].aggregate(["mean", "min", "max"]))
    return


@app.cell
def _(df_inverse):
    df_inverse.groupby("clusters")["Age"].aggregate(["mean", "min", "max"])
    return


@app.cell
def _(df_inverse):
    df_inverse.groupby("clusters")["Education"].aggregate(["mean", "min", "max"])
    return


@app.cell
def _(df_inverse, plt, sns):
    sns.pairplot(df_inverse, hue='clusters', vars=['Income', 'Age', 'Education'])
    plt.show()
    return


@app.cell
def _(df_inverse, plt, sns):
    plt.figure(figsize=(25,8))
    sns.scatterplot(df_inverse, x="Income", y="Age", hue="clusters", palette="Set1")
    plt.title("Age vs Income")
    plt.show()
    return


@app.cell
def _(df_inverse, plt, sns):
    plt.figure(figsize=(25,8))
    sns.scatterplot(df_inverse, x="Income", y="Education", hue="clusters", palette="Set1")
    plt.title("Age vs Income")
    plt.show()
    return


@app.cell
def _(df_inverse, plt, sns):
    plt.figure(figsize=(25,8))
    sns.swarmplot(df_inverse, y="Income", x="Education", hue="clusters", size=4.5, palette='Set1')
    plt.title("Education vs Income")
    plt.show()
    return


@app.cell
def _(df_inverse):
    print(df_inverse.groupby("clusters")[["Income", "Age", "Education"]].aggregate("mean"))
    print(df_inverse["clusters"].value_counts())
    return


@app.cell
def _(df_inverse):
    print(df_inverse.groupby("clusters")["Education"].aggregate("value_counts")[4])
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    # Intepretation 

    From the above analysis we note the following:

    |Clustering|Total Count|Income (Mean) |Age (Mean) |Education (Count)|
    |:---:|:---|:---|:---|:---|
    |0 |293|Lowest (70k) |Young 20~40 (30) | Mostly high school (~200) |
    |1 |334|Highest (170k)|Young 20~40 (33)|  Mostly high school (~250)|
    |2 |234|High (150k)|High > 40 (60)| Mostly university (~180) |
    |3 |442|Mid (110k)|Mid >30 (42)| Mostly highschool (~350) |
    |4 |697|Mid (110)|Young < 30|Mostly highschool (~550)|

    # Marketing Strategies 
        
    ## Cluster 0
    - **Profile:** Young adult, low income, and mostly high school education.
    - **Strategy:** We should offer this group products that are low cost and essential. It is important to show the good value and make them easy to find on social media and mobile phones.

    ## Cluster 1
    - **Profile:** Highest income, young, mostly high school gradute):
    - **Strategy:** For this group, we can promote products that are new, fashionable, and offer good experiences. We should use online influencers and digital places where rich young people spend time.

    ## Cluster 2
    - **Profile:** High income, high age, mostly university gradute
    - **Strategy:** We can offer these customers very high-quality products or services. We should focus on comfort, how reliable they are, and benefits for a long time, using ads where professionals look.

    ## Cluster 3
    - **Profile:** Mid income, mid age, mostly high school
    - **Strategy:** We should market useful products for families or for improving life, at a medium price. We need to show they are good value and dependable, with messages they can connect to.

    ## Cluster 4
    - **Profile:** Mid income, young, mostly high school
    - **Strategy:** We can attract this group with promotions for popular items and fun activities that are not too expensive. We must focus on social media to connect with them and create a community.


    """
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()

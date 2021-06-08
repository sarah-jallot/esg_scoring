<!-- #region -->
# Data exploration, clustering, and ESG Score prediction. 
In this notebook, our reference universe will correspond to all authorized stocks within the **Natixis Investment Managers Challenge, 2021 edition**. It comprises over **2,821 companies** for which we retrieved ESG data for year 2020. 
  

 
Our goal is to use **Sustainable Finance Disclosure Regulation (SFDR)** metrics to predict company ESG performance.  
We will do this in three ways: 
- By performing clustering on these companies based on SFDR metrics, and analysing our results. 
- By predicting the companies' ESG Score according to three data providers: Thompson-Reuters, MSCI, and CSRHub.  
- Ultimately, we aim to build our own robust & transaparent ESG Score based on our research work. 
<!-- #endregion -->

First, run all necessary imports.


# Imports

```python
import pandas as pd
import pprint
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from numpy import unique
from numpy import where
import seaborn as sns
import pprint
from sklearn.preprocessing import OneHotEncoder
from pprint import PrettyPrinter

import sys
sys.path.append("../utils/")
from utils import *

from sklearn.preprocessing import StandardScaler
#Importing required modules
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from sklearn.ensemble import RandomForestClassifier
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import confusion_matrix, accuracy_score

# Import clustering methods
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import Birch
from sklearn.cluster import SpectralClustering
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.cluster import MeanShift
from sklearn.cluster import OPTICS
```

First, download your data.

```python
input_path = "../inputs/"
input_filename = "universe_df_esg.csv"
output_path = "../outputs/"
initial_df = pd.read_csv(input_path+input_filename).drop(columns=["Unnamed: 0"])
df = initial_df.loc[:,"Name":"Bribery, Corruption and Fraud Controversies"].copy()
df["ESG Score"] = initial_df["ESG Score"]
df['Environmental Pillar Score'] = initial_df.loc[:,"Environmental Pillar Score"]
df['Social Pillar Score'] = initial_df.loc[:,"Social Pillar Score"]
df['Governance Pillar Score'] = initial_df.loc[:,"Governance Pillar Score"]
df["ESG Score Grade"] = initial_df["ESG Score Grade"]
df['Environmental Pillar Score Grade'] = initial_df.loc[:,"Environmental Pillar Score Grade"]
df['Social Pillar Score Grade'] = initial_df.loc[:,"Social Pillar Score Grade"]
df['Governance Pillar Score Grade'] = initial_df.loc[:,"Governance Pillar Score Grade"]
prep_df = pd.read_csv("../inputs/universe_df_encoded.csv")
```

```python
# And the mapping of each metric to a pillar
pillar_mapping = {
    # Environmental
    'CO2 Equivalent Emissions Total': 'Environmental',
    'CO2 Equivalent Emissions Direct, Scope 1': 'Environmental',
    'CO2 Equivalent Emissions Indirect, Scope 2': 'Environmental',
    'CO2 Equivalent Emissions Indirect, Scope 3':'Environmental',
    'Total CO2 Equivalent Emissions To Revenues USD in million':'Environmental',
    'Total Renewable Energy To Energy Use in million': "Environmental",
    'Total Energy Use To Revenues USD in million': "Environmental",
    'Biodiversity Impact Reduction': "Environmental",
    'Water Pollutant Emissions To Revenues USD in million': "Environmental",
    'Hazardous Waste': "Environmental", 
    'Waste Total': "Environmental", 
    'Waste Recycled Total': "Environmental",
    
    # Social
    'Salary Gap':"Social",
    'Accidents Total':"Social",
    'Critical Country 1':"Social",
    'Critical Country 2':"Social", 
    'Critical Country 3':"Social", 
    'Critical Country 4':"Social",
    'Board Gender Diversity, Percent':"Social", 
    
    # Governance
    'Whistleblower Protection': "Governance",
    'Fundamental Human Rights ILO UN': "Governance", 
    'Human Rights Policy': "Governance", 
    'Anti-Personnel Landmines':"Governance",
    'Bribery, Corruption and Fraud Controversies':"Governance", 
    
    # Target
    'ESG Score Grade':"Target",
    'Environmental Pillar Score Grade':"Target", 
    'Social Pillar Score Grade':"Target",
    'Governance Pillar Score Grade':"Target", 
    'Environmental Innovation Score Grade':"Target",
    'CSR Strategy Score Grade':"Target", 
    'ESG Score':"Target", 
    'Environmental Pillar Score':"Target",
    'Social Pillar Score':"Target", 
    'Governance Pillar Score':"Target",
    
    # Other
    'Total CO2 Equivalent Emissions To Revenues USD Score':"Other",
    'Environmental Innovation Score':"Other", 
    'CSR Strategy Score':"Other",
    'Market Capitalization (bil) [USD]': 'Other',
    'Name':'Other',
    'Symbol':'Other', 
    'Country':'Other', 
    'Industry Name - GICS Sub-Industry':'Other', 
    'SEDOL':'Other', 
    'ISINS':'Other',
    'GICS Sector Name':'Other', 
    'NACE Classification':'Other',
}
```

# Basic Clustering


## Dimensionality reduction techniques


Note that PCA works for data that is linearly separable. Given the complexity of our data, this may well not be the case as we will explore here.  
First, we drop non-numerical columns.

```python
drop_cols = [
    "Name",
    "Symbol",
    "Country",
 #   "Industry Name - GICS Sub-Industry",
    "SEDOL",
    "ISINS",
    "GICS Sector Name",
    "Critical Country 1",
    "Critical Country 2",
    "Critical Country 3",
    "Critical Country 4",
    "ESG Score",
    "Environmental Pillar Score", 
    "Social Pillar Score", 
    "Governance Pillar Score",
    "ESG Score Grade",
    "Environmental Pillar Score Grade", 
    "Social Pillar Score Grade", 
    "Governance Pillar Score Grade",
]
cluster_df = prep_df.drop(columns=drop_cols).copy()
print(cluster_df.shape)
cluster_df.head()
```

```python
cluster_df.to_csv("../inputs/universe_clusters.csv", index=False)
```

#### a) PCA

```python
threshold = 0.8
method = "cumsum"
```

```python
full_pca_features, pca, selected_features = pca_selection(
    cluster_df, 
    n_components=cluster_df.shape[1], 
    threshold=threshold, 
    method=method
)
print(f"We selected {selected_features.shape[1]} out of {full_pca_features.shape[1]} features for the {method} method with a threshold of {threshold} .")
```

```python
plot_pca(pca,percent=False, cumsum=False)
```

```python
plot_pca(pca,percent=True, cumsum=True)
```

We can check visually that there is no jump in explained variance, meaning that our PCA isn't very useful in reducing data dimension given its complexity. 

```python
prep_df["PCA_1"] = pd.Series(full_pca_features[:,0])
prep_df["PCA_2"] = pd.Series(full_pca_features[:,1])
scatterplot(prep_df, x_axis="PCA_1", y_axis="PCA_2", title="PCA scatterplot", hue=None, filename="pca_scatterplot.png")
```

```python
scatterplot(prep_df, x_axis="PCA_1", y_axis="PCA_2", title="PCA scatterplot by sector", hue="GICS Sector Name", filename="pca_scatterplot_coloured.png")
```

```python
X_pca = selected_features
pd.DataFrame(X_pca).to_csv("../inputs/X_pca.csv", index=False)
```

### Kernel PCA 
Applying a kernel to make the data linearly separable by PCA. 

```python
threshold = 0.8
n_components = cluster_df.shape[1]
kernel = "rbf"
method = "cumsum"
```

```python
full_kpca_features, kpca, selected_kpca_features = kpca_selection(cluster_df, n_components=n_components, kernel=kernel, threshold=threshold, method=method)
print(f"We selected {selected_kpca_features.shape[1]} out of {full_kpca_features.shape[1]} features for the {method} method with a threshold of {threshold} .")
```

```python
prep_df["KPCA_1"] = pd.Series(full_kpca_features[:,0])
prep_df["KPCA_2"] = pd.Series(full_kpca_features[:,1])
```

```python
plot_kpca(kpca, full_kpca_features, percent=False, cumsum=False)
```

```python
plot_kpca(kpca, full_kpca_features, percent=True, cumsum=True)
```

In this graph, we see that KPCA is more successful at reducing data dimensionality, although we are not in an ideal setting. 

```python
plt.scatter(x=full_kpca_features[:,0], y=full_kpca_features[:,1])
plt.title("Kernel PCA feature scatterplot")
plt.show()
```

```python
scatterplot(
    prep_df, 
    x_axis="KPCA_1", 
    y_axis="KPCA_2", 
    title="KPCA scatterplot by sector", 
    hue="GICS Sector Name", 
    filename="kpca_sector_scatterplot.png"
)
```

```python
X_kpca = selected_kpca_features
pd.DataFrame(X_kpca).to_csv("../inputs/X_kpca.csv", index=False)
prep_df.to_csv("../inputs/universe_df_encoded_pca.csv", index=False)
```

### Random Forest


We train a Random Forest to predict the ESG Score as a function of our SFDR metrics, and use a threshold feature importance to select relevant features.  

```python
threshold = 0.005
X, y = cluster_df.copy(), prep_df["ESG Score"].copy()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)
```

```python
importances, masked_importances, r2_full, r2_imp = random_forest_selection(
    X_train, 
    X_test, 
    y_train, 
    y_test, 
    threshold=threshold
)
print(f"Full R2 score: {r2_full:.2f} versus score after feature selection: {r2_imp:.2f}.")
```

Observations: 
- The two most important features to predict Refinitiv ESG scoring are **governance related** and concern the implementation of a **Human Rights Policy** within the organization.  
- **Market Capitalization** comes third. This could point to a bias within the data towards capitalization. We know that bigger companies have more resources dedicated to transparency, and that on material issues, Refinitiv applies transparency weights. They might be biased by capitalization even with their corrections.  
- We observe several occurrences of **CO2 Equivalent Emissions** as a total and by scope 1, 2 and 3, as well as CO2 intensity. It could be interesting to keep only one or two of these variables as they seem close (to be confirmed).  
- Some variables within our best predictors presented limited coverage : Hazardous Waste for example has 70% missing values.


Let's have a look at the countplot within the initial dataframe for Fundamental Human Rights policies:

```python
countplot(df,category="Fundamental Human Rights ILO UN", filename="human_rights.png", )
print(df["Fundamental Human Rights ILO UN"].value_counts()/df["Fundamental Human Rights ILO UN"].value_counts().sum())
```

```python
df.groupby("Fundamental Human Rights ILO UN").mean().loc[:,["Market Capitalization (bil) [USD]", "ESG Score"]]
```

```python
columns = ["ESG Score Grade", "GICS Sector Name", "Fundamental Human Rights ILO UN"]
catplot(df, columns, figsize=(11,0.5),order=['A+', 'A', 'A-', 'B+', 'B', 'B-', 'C+', 'C', 'C-', 'D+', 'D', 'D-'])
```

It would seem that only 34% of our investees have signed the Fundamental Human Rights ILO UN Convention.

```python
upper = 10
mask = df["Market Capitalization (bil) [USD]"] < upper
scatterplot(
    df[mask], 
    x_axis="Market Capitalization (bil) [USD]", 
    y_axis="ESG Score", 
    title="ESG Score as a function of Market Cap",
    hue=None,
    filename="esg_score_market_cap_scatterplot.png"
)
```

```python
scatterplot(
    df, 
    x_axis="Board Gender Diversity, Percent", 
    y_axis="ESG Score", 
    title="ESG Score as a function of Board gender diversity, %",
    hue=None,
    filename="esg_score_gender_div_scatterplot.png"
)

```

```python
mask = df["Salary Gap"] < 500

scatterplot(
    df[mask], 
    x_axis="Salary Gap", 
    y_axis="ESG Score", 
    title="ESG Score as a function of Salary Gap",
    hue=None,
    filename="esg_score_salary_gap_scatterplot.png"
)
```

Let's have a look at a scatterplot for an environmental indicator: 

```python
scatterplot(
    df, 
    x_axis="CO2 Equivalent Emissions Indirect, Scope 2", 
    y_axis="ESG Score", 
    title="ESG Score as a function of Scope 2 Carbon Emissions",
    hue=None,
    filename="esg_score_co2_eq_scope2_scatterplot.png"
)
```

We can assume that the best-in-class notation makes CO2 emissions a rather bad overall predictor, as it is mostly linked to the industry. 

```python
mask = importances.importances > threshold
X_rf = cluster_df.loc[:,importances[mask].features]
pd.DataFrame(X_rf).to_csv("../inputs/X_rf.csv", index=False)
X_rf_target = X_rf.copy()
X_rf_target["ESG Score"] = prep_df["ESG Score"].copy()
X_rf_target.head()
```

We plot correlations for X_rf including the target variable, our ESG Score.

```python
corrplot(
    X_rf_target, 
    filename = "corrplot_random_forest.png",
    vmax=1, 
    title="Correlation matrix for Random Forest selected features"
)
```

Coming from the USA is slightly negatively correlated to ESG Scoring. We could infer that American companies perform less well than European ones according to Refinitiv, although they are overrepresented in our dataset.

```python
pillars = [pillar_mapping.get(key) for key in X_rf.columns]
pillars = list(filter(lambda a: a != "Other", pillars))
pillars = list(filter(lambda a: a != "Target", pillars))
sns.countplot(pillars, order=dict(Counter(pillars).most_common()).keys())
plt.title("Pillar count for most important features in our Random Forest.")
plt.show()

for key in dict(Counter(pillars)).keys():
    print(f"{key}: {dict(Counter(pillars))[key]/sum(list(Counter(pillars).values()))*100:.1f} percent")
```

Environmental metrics are over-represented within our dataset versus original metrics. We see that coming from the USA is used as a predictor in ESG Scoring, a relationship we will investigate.

```python
scatterplot(
    X_rf, 
    x_axis="Board Gender Diversity, Percent", 
    y_axis="Market Capitalization (bil) [USD]", 
    title="Random Forest Feature Selection",
    hue=None,
    filename="market_cap_gender_div_scatterplot.png"
)
```

### Autoencoder

```python
# to do 
```

### Clustering


As such, cluster analysis is an iterative process where subjective evaluation of the identified clusters is fed back into changes to algorithm configuration until a desired or appropriate result is achieved.


Our dimensionality reduction process pushes us to retain the 18 features from the Random Forest Algorithm.

```python
X_rf = pd.read_csv("../inputs/X_rf.csv")
prep_df = pd.read_csv("../inputs/universe_df_encoded_pca.csv")
prep_df["ESG Category"] = simplify_categories(prep_df["ESG Score Grade"])
df = pd.read_csv("../inputs/universe_df_no_nans.csv").drop(columns=["Unnamed: 0"])
```

```python
print(f"{X_rf.shape[1]} features:  {list(X_rf.columns)}")
```

```python
#X = np.array(pd.read_csv("../inputs/X_pca.csv"))
#X = np.array(pd.read_csv("../inputs/X_kpca.csv"))
scaler = StandardScaler()
X = scaler.fit_transform(np.array(X_rf))
```

#### K-means


Assign examples to each cluster while trying to minimize the variance within each cluster. Let's perform the elbow method to determine the optimal number of clusters.


We retro-ajust the number of clusters to make them informative regarding ESG Scoring. 

```python
kmeans_kwargs = {
    "init": "random",
    "n_init": 10,
    "max_iter": 300,
    "random_state": 42,
}
```

```python
def kmeans_cramer(X, kmeans_kwargs, categories, upper=10, plot=True, corrected=False):
    """
    Run the kmeans algorithm for various numbers of clusters.
    Plot the elbow graph to find the optimal k.
    X: normalised features to perform clustering on.
    kmeans_kwargs: dictionary containing your kmeans arguments.
    upper: the maximal number of clusters to test.
    plot: boolean indicating whether to plot the elbow graph.

    :returns: the sse as a list.
    """
    # A list holds the Cramers values for each k
    cramers = []
    cramers_corrected = []
    lower = 1
    for k in range(lower, upper):
        kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
        kmeans.fit(X)
        df = pd.DataFrame(categories)
        df["labels"] = kmeans.labels_
        df.columns = ["ESG Category","kmean_labels"]
        cramers.append(cramers_stat(df, "kmean_labels", 'ESG Category'))
        cramers_corrected.append(cramers_stat(df, "kmean_labels", 'ESG Category'))
    if plot == True:
        plt.style.use("fivethirtyeight")
        plt.figure(figsize=(15, 5))
        if corrected == True:
            plt.plot(range(lower + 1, upper + 1), cramers_corrected)
            plt.title("Elbow graph for KMeans - Cramers Corrected")
        else:
            plt.plot(range(lower + 1, upper + 1), cramers)
            plt.title("Elbow graph for KMeans - Cramers")
        plt.xticks(range(lower + 1, upper + 1), rotation=45)
        plt.xlabel("Number of Clusters")
        plt.ylabel("Cramers")
        plt.savefig('images/elbow_graph_kmeans_cramers.png')
        plt.show()
    return cramers


def m_kmeans_cramer(X, categories, upper=10, plot=True, corrected=False):
    """
    Run the kmeans algorithm for various numbers of clusters.
    Plot the elbow graph to find the optimal k.
    X: normalised features to perform clustering on.
    kmeans_kwargs: dictionary containing your kmeans arguments.
    upper: the maximal number of clusters to test.
    plot: boolean indicating whether to plot the elbow graph.

    :returns: the sse as a list.
    """
    # A list holds the Cramers values for each k
    cramers = []
    cramers_corrected = []
    lower = 1
    for k in range(lower, upper):
        model = MiniBatchKMeans(n_clusters=k)
        model.fit(X)
        df = pd.DataFrame(categories)
        df["labels"] = model.labels_
        df.columns = ["ESG Category","mkmean_labels"]
        cramers.append(cramers_stat(df, "mkmean_labels", 'ESG Category'))
        cramers_corrected.append(cramers_corrected_stat(df, "mkmean_labels", 'ESG Category'))
    if plot == True:
        plt.style.use("fivethirtyeight")
        plt.figure(figsize=(15, 5))
        if corrected == True:
            plt.plot(range(lower + 1, upper + 1), cramers_corrected)
            plt.title("Elbow graph for MKMeans - Cramers Corrected")
        else:
            plt.plot(range(lower + 1, upper + 1), cramers)
            plt.title("Elbow graph for MKMeans - Cramers")
        plt.xticks(range(lower + 1, upper + 1))
        plt.xlabel("Number of Clusters")
        plt.ylabel("SSE")
        plt.title("Elbow graph for Mini-batch KMeans")
        plt.show()
        plt.savefig('images/elbow_graph_mkmeans_cramer.png')
        plt.show()
    return cramers
```

```python
sse = kmeans(X, kmeans_kwargs, categories=prep_df["ESG Category"], upper=50, plot=True)
```

```python
cramers = kmeans_cramer(X, kmeans_kwargs, prep_df["ESG Category"], upper=50, plot=True, corrected=False)
```

```python
optimal_nb = 2
optimal_kmeans = KMeans(n_clusters=optimal_nb, **kmeans_kwargs)
optimal_kmeans.fit(X)
```

```python
prep_df["kmean_labels"] = optimal_kmeans.labels_
X_rf["kmean_labels"] = optimal_kmeans.labels_
```

Now, let's see whether our clusters seem to separate companies efficiently along ESG Scoring: 

```python
print(f"Cramer's coeff: {cramers_stat(prep_df, 'kmean_labels', 'ESG Category'):.2f} | Corrected coeff: {cramers_corrected_stat(prep_df, 'kmean_labels', 'ESG Category'):.2f}")
```

```python
categorical_countplot(prep_df, "kmean_labels", "ESG Category", filename="ESG_cat_kmeans.png")
```

```python
scatterplot(
    prep_df, 
    "KPCA_1", 
    "KPCA_2", 
    "kmean_labels", 
    "KPCA coloured by kmeans",
    "kpca_kmeans_scatterplot.png"
)
```

#### Mini-batch Kmeans

```python
sse = m_kmeans(X, upper=50, plot=True)
```

```python
cramers = m_kmeans_cramer(X, prep_df["ESG Category"], upper=50, plot=True, corrected=False)
```

```python
optimal_nb = 15
optimal_mkmeans = MiniBatchKMeans(n_clusters=optimal_nb)
optimal_mkmeans.fit(X)
```

```python
prep_df["mkmean_labels"] = optimal_mkmeans.labels_
X_rf["mkmean_labels"] = optimal_mkmeans.labels_
```

```python
prep_df["mkmean_labels"].value_counts()
```

```python
print(f"Cramer's coeff: {cramers_stat(prep_df, 'mkmean_labels', 'ESG Category'):.2f} | Corrected coeff: {cramers_corrected_stat(prep_df, 'mkmean_labels', 'ESG Category'):.2f}")
```

```python
categorical_countplot(prep_df, "mkmean_labels", "ESG Category", ["A","B","C","D"], filename="ESG_cat_mkmeans.png")
```

```python
scatterplot(
    prep_df, 
    "KPCA_1", 
    "KPCA_2", 
    "mkmean_labels", 
    "KPCA coloured by Mini-kmeans",
    "kpca_kmeans_scatterplot.png"
)
```

#### Affinity propagation


Affinity propagation finds "exemplars," members of the input set that are representative of clusters.

```python
damping = 0.7
```

```python
# define the model
aff_model = AffinityPropagation(damping=damping)
# fit the model
aff_model.fit(X)
```

```python
prep_df["aff_labels"] = aff_model.labels_
X_rf["aff_labels"] = aff_model.labels_
```

```python
X_rf["aff_labels"].value_counts()
```

```python
print(f"Cramer's coeff: {cramers_stat(prep_df, 'aff_labels', 'ESG Category'):.2f} | Corrected coeff: {cramers_corrected_stat(prep_df, 'aff_labels', 'ESG Category'):.2f}")
```

```python
scatterplot(
    prep_df, 
    "KPCA_1", 
    "KPCA_2", 
    "aff_labels", 
    "KPCA coloured by Affinity propagation",
    "kpca_affinity_scatterplot.png"
)
```

```python
categorical_countplot(
    prep_df, 
    "aff_labels", 
    "ESG Category", 
    ["A","B","C","D"], 
    filename="ESG_cat_affinity.png")
```

#### Agglomerative clustering


Belongs to hierarchical clustering methods.

```python
n_clusters = 10
```

```python
# define the model
agg_model = AgglomerativeClustering(n_clusters=n_clusters)
# fit model and predict clusters
yhat = agg_model.fit_predict(X)
```

```python
prep_df["agg_labels"] = agg_model.labels_
X_rf["agg_labels"] = agg_model.labels_
```

```python
X_rf["agg_labels"].value_counts()
```

```python
scatterplot(
    prep_df, 
    "KPCA_1", 
    "KPCA_2", 
    "agg_labels", 
    "KPCA coloured by Aggregation",
    "kpca_aggregation_scatterplot.png"
)
```

```python
print(f"Cramer's coeff: {cramers_stat(prep_df, 'agg_labels', 'ESG Category'):.2f} | Corrected coeff: {cramers_corrected_stat(prep_df, 'agg_labels', 'ESG Category'):.2f}")
```

```python
categorical_countplot(
    prep_df, 
    "agg_labels", 
    "ESG Category", 
    ["A","B","C","D"], 
    filename="ESG_cat_aggregation.png")
```

#### BIRCH Clustering


Constructing a tree structure from which cluster centroids are extracted.

```python
threshold = 0.08
```

```python
# define the model
birch_model = Birch(threshold=threshold, n_clusters=optimal_nb)
# fit the model
yhat = birch_model.fit_predict(X)
```

```python
prep_df["birch_labels"] = birch_model.labels_
X_rf["birch_labels"] = birch_model.labels_
```

```python
X_rf["birch_labels"].value_counts()
```

```python
scatterplot(
    prep_df, 
    "KPCA_1", 
    "KPCA_2", 
    "birch_labels", 
    "KPCA coloured by Mini-kmeans",
    "kpca_birch_scatterplot.png"
)
```

```python
print(f"Cramer's coeff: {cramers_stat(prep_df, 'birch_labels', 'ESG Category'):.2f} | Corrected coeff: {cramers_corrected_stat(prep_df, 'birch_labels', 'ESG Category'):.2f}")
```

```python
categorical_countplot(
    prep_df, 
    "birch_labels", 
    "ESG Category", 
    ["A","B","C","D"], 
    filename="ESG_cat_affinity.png")
```

#### DBSCAN clustering


Used to find clusters of arbitrary shape.

```python
eps = 0.30
min_samples =50
```

```python
# define the model
db_model = DBSCAN(eps=eps, min_samples=min_samples)
# fit model and predict clusters
yhat = db_model.fit_predict(X)
```

```python
prep_df["dbscan_labels"] = db_model.labels_
X_rf["dbscan_labels"] = db_model.labels_
```

```python
X_rf["dbscan_labels"].value_counts()
```

```python
print(f"Cramer's coeff: {cramers_stat(prep_df, 'dbscan_labels', 'ESG Category'):.2f} | Corrected coeff: {cramers_corrected_stat(prep_df, 'dbscan_labels', 'ESG Category'):.2f}")
```

```python
scatterplot(
    prep_df, 
    "KPCA_1", 
    "KPCA_2", 
    "dbscan_labels", 
    "KPCA coloured by DBSCAN",
    "kpca_dbscan_scatterplot.png"
)
```

```python
categorical_countplot(
    prep_df, 
    "dbscan_labels", 
    "ESG Category", 
    ["A","B","C","D"], 
    filename="ESG_cat_dbscan.png")
```

#### Mean Shift clustering


Mean shift clustering involves finding and adapting centroids based on the density of examples in the feature space.

```python
# define the model
mshift_model = MeanShift(min_bin_freq=50)
# fit model and predict clusters
yhat = mshift_model.fit_predict(X)
```

```python
prep_df["mshift_labels"] = mshift_model.labels_
X_rf["mshift_labels"] = mshift_model.labels_
```

```python
X_rf["mshift_labels"].value_counts()
```

```python
scatterplot(
    prep_df, 
    "KPCA_1", 
    "KPCA_2", 
    "mshift_labels", 
    "KPCA coloured by Mean-shift",
    "kpca_mshift_scatterplot.png"
)
```

```python
print(f"Cramer's coeff: {cramers_stat(prep_df, 'mshift_labels', 'ESG Category'):.2f} | Corrected coeff: {cramers_corrected_stat(prep_df, 'mshift_labels', 'ESG Category'):.2f}")
```

```python
categorical_countplot(
    prep_df, 
    "mshift_labels", 
    "ESG Category", 
    ["A","B","C","D"], 
    filename="ESG_cat_mshift.png")
```

#### OPTICS


Modified version of DBSCAN. 

```python
eps = eps
min_samples = min_samples
```

```python
optics_model = OPTICS(eps=eps, min_samples=min_samples)
# fit model and predict clusters
yhat = optics_model.fit_predict(X)
```

```python
prep_df["optics_labels"] = optics_model.labels_
X_rf["optics_labels"] = optics_model.labels_
```

```python
X_rf["optics_labels"].value_counts()
```

```python
scatterplot(
    prep_df, 
    "KPCA_1", 
    "KPCA_2", 
    "optics_labels", 
    "KPCA coloured by Optics",
    "kpca_optics_scatterplot.png"
)
```

```python
print(f"Cramer's coeff: {cramers_stat(prep_df, 'optics_labels', 'ESG Category'):.2f} | Corrected coeff: {cramers_corrected_stat(prep_df, 'optics_labels', 'ESG Category'):.2f}")
```

```python
categorical_countplot(
    prep_df, 
    "optics_labels", 
    "ESG Category", 
    ["A","B","C","D"], 
    filename="ESG_cat_optics.png")
```

#### Spectral clustering

```python
n_clusters = 20
```

Here, one uses the top eigenvectors of a matrix derived from the distance between points.

```python
# define the model
spec_model = SpectralClustering(n_clusters=n_clusters)
# fit model and predict clusters
yhat = spec_model.fit_predict(X)
```

```python
prep_df["spectral_labels"] = spec_model.labels_
X_rf["spectral_labels"] = spec_model.labels_
```

```python
scatterplot(
    prep_df, 
    "KPCA_1", 
    "KPCA_2", 
    "spectral_labels", 
    "KPCA coloured by Spectral labels",
    "kpca_spectral_scatterplot.png"
)
```

```python
print(f"Cramer's coeff: {cramers_stat(prep_df, 'spectral_labels', 'ESG Category'):.2f} | Corrected coeff: {cramers_corrected_stat(prep_df, 'spectral_labels', 'ESG Category'):.2f}")
```

```python
categorical_countplot(
    prep_df, 
    "spectral_labels", 
    "ESG Category", 
    ["A","B","C","D"], 
    filename="ESG_cat_spectral.png")
```

#### Gaussian Mixture Clustering


A Gaussian mixture model summarizes a multivariate probability density function with a mixture of Gaussian probability distributions as its name suggests.

```python
n_clusters = 20
```

```python
# define the model
gmm_model = GaussianMixture(n_components=n_clusters)
# fit the model
gmm_model.fit(X)
```

```python
prep_df["gaussian_labels"] = gmm_model.predict(X)
X_rf["gaussian_labels"] = gmm_model.predict(X)
```

```python
X_rf["gaussian_labels"].value_counts()
```

```python
scatterplot(
    prep_df, 
    "KPCA_1", 
    "KPCA_2", 
    "gaussian_labels", 
    "KPCA coloured by Gaussian Labels",
    "kpca_gaussian_scatterplot.png"
)
```

```python
print(f"Cramer's coeff: {cramers_stat(prep_df, 'gaussian_labels', 'ESG Category'):.2f} | Corrected coeff: {cramers_corrected_stat(prep_df, 'gaussian_labels', 'ESG Category'):.2f}")
```

```python
categorical_countplot(
    prep_df, 
    "gaussian_labels", 
    "ESG Category", 
    ["A","B","C","D"], 
    filename="ESG_cat_gaussian.png")
```

```python
X_rf.to_csv("../inputs/X_rf_labelled.csv", index=False)
prep_df.to_csv("../inputs/prep_df_labelled.csv", index=False)
```

### Cluster interpretation and visualisation. 


Let us now examine our variable within our different clusters.

```python
df = pd.read_csv("../inputs/prep_df_labelled.csv")
df.head()
```

```python
columns = ["ESG Score Grade", "kmean_labels", "Fundamental Human Rights ILO UN"]
catplot(df, columns, figsize=(11,0.5))
```

```python
plt.figure(figsize=(10,10))
sns.catplot(x="kmean_labels", y="ESG Score", hue="Fundamental Human Rights ILO UN", kind="box", data=df)
plt.show()
```

```python
# count the label distribution by gics sector
```

```python
columns = ["kmean_labels","Board Gender Diversity, Percent"]
boxplot(df, columns, filename="gender_div_kmeans.png", categorical=True, figsize=(10,6))
```

```python
columns = ["kmean_labels","Total CO2 Equivalent Emissions To Revenues USD in million"]
boxplot(df, columns, filename="co2_emissions_kmeans.png", categorical=True, figsize=(10,6))
```

```python
columns = ["ESG Score Grade", "kmean_labels", "GICS Sector Name"]
catplot(df, columns, figsize=(11,0.5))
```

```python
columns = ["ESG Score Grade", "kmean_labels", "Fundamental Human Rights ILO UN"]
catplot(df, columns, figsize=(11,0.5))
```

```python
continuous_name = "Board Gender Diversity, Percent"
# 'Market Capitalization (bil) [USD]','CO2 Equivalent Emissions Indirect, Scope 2', 'Salary Gap',
# 'Accidents Total', 'CO2 Equivalent Emissions Direct, Scope 1',
# 'CO2 Equivalent Emissions Total', 'Hazardous Waste',
# 'CO2 Equivalent Emissions Indirect, Scope 3',
# 'Total Renewable Energy To Energy Use in million','Total CO2 Equivalent Emissions To Revenues USD in million',
#'Total Energy Use To Revenues USD in million', 'Waste Total',
#'Waste Recycled Total', 
# 'CO2 Equivalent Emissions Total'
categorical_name = "Fundamental Human Rights ILO UN"
#'Human Rights Policy','Biodiversity Impact Reduction','Whistleblower Protection',

clusters = "kmean_labels"
```

```python
cluster_boxplot(df, feature_name = continuous_name, clusters=clusters)
```

```python
cluster_catplot(df, feature_name =categorical_name , clusters=clusters)
```

```python
df["ESG Category"] = simplify_categories(df["ESG Score Grade"])
```

```python
categorical_countplot(
    df, 
    "mkmean_labels", 
    "ESG Category", 
    ["A","B","C","D"], 
    filename="ESG_cat_mkmeans.png")
```

```python
print(f"Cramer's coeff: {cramers_stat(df, 'mkmean_labels', 'ESG Category'):.2f} | Corrected coeff: {cramers_corrected_stat(df, 'mkmean_labels', 'ESG Category'):.2f}")
```

Once we have made satisfactory clusters with our data, we are going to use them first to predict Refinitiv score and then to build our own.


## Refinitiv score prediction


L'idée est de voir si, à partir des indicateurs requis par SFDR, il nous est possible de retrouver les scores ESG de Refinitiv.


**L'esprit de la notation ESG Refinitiv:**
- Performance relative de l'entreprise par rapport à son secteur d'activité sur le E et le S. 
- Performance relative au pays d'implantation sur le G. 

**La méthodologie de notation ESG Refinitiv:**  
- Sous-sélection de 186/500 indicateurs comparables
- Mapping sur chaque indicateur du degré de matérialité d'une problématique sur une échelle de 1 à 10 
- Transparence très importante sur les métriques définies comme hautement matérielles  
- Analyse de controverse, avec correction du biais contre les grosses market cap


**Par pilier, leur méthodologie varie.**
- Sur le pilier environnemental, ils prennent d'habitude une médiane par industrie.  
- Sur le pilier social, ils appliquent la matrice de transparence et pour le %d'employées et le % de représentation, la médiane par industrie.
- Sur le pilier de la gouvernance, ils regardent l'ensemble des indicateurs.  


[Random Forest finetuning](https://towardsdatascience.com/random-forest-hyperparameters-and-how-to-fine-tune-them-17aee785ee0d)


### ESG Score Classification

```python
X = pd.read_csv(input_path+"X_rf_labelled.csv")
y = pd.read_csv(input_path+"/universe_df_encoded.csv").loc[:,"ESG Score Grade"]
df = X.copy()
df["ESG Score Grade"] = y.copy()
df["ESG Category"] = simplify_categories(df["ESG Score Grade"])
```

```python
y = simplify_categories(y)
```

```python
params = {
    "n_estimators":200, 
    "max_depth":10,
}
```

```python
model, X_train, X_test, y_train, y_test = train_random_forest(X, y, params, test_size=0.2)
```

```python
y_pred = model.predict(X_test)
```

```python
confusion_mat_df(model, y_test, y_pred, percent=False)
```

```python
confusion_mat_df(model, y_test, y_pred, percent=True)
```

```python
pp = PrettyPrinter(indent=3)
pp.pprint(list(X.columns[-10:]))
```

```python
labels = list(X.columns[-10:])
for label in labels:
    print(f"Cramer's coeff {label} : {cramers_stat(df, label, 'ESG Category'):.2f} | Corrected coeff {label} : {cramers_corrected_stat(df, label, 'ESG Category'):.2f}")
```

```python
model, X_train, X_test, y_train, y_test = train_random_forest(
    X, 
    y, 
    params, 
    test_size=0.2, 
    with_labels=True, 
    labels="mkmean_labels",)
```

```python
y_pred = model.predict(X_test)
```

```python
confusion_mat_df(model, y_test, y_pred, percent=False)
```

```python
confusion_mat_df(model, y_test, y_pred, percent=True)
```

From this confusion matrix, we see that category A is predicted very well from SFDR metrics + mkmeans labels.

```python
best_model = model
```

```python
n_features = 15
importances = pd.DataFrame(list(X_train.columns))
importances["feature_importances"] = best_model.feature_importances_
importances.columns = ["features", "importances"]
importances = importances.sort_values(by=["importances"], ascending=False).reset_index().copy()
importances.loc[:, "features":"importances"].plot(kind="barh")
plt.yticks(ticks=range(n_features), labels=importances[:n_features]["features"])
plt.show()
```

### MSCI Prediction

```python
initial_df = pd.read_csv("../inputs/universe_df_encoded_pca.csv")
df = add_msci(initial_df)
df.shape
```

```python
df = df.drop_duplicates(["Name"])
```

```python
X_temp = pd.read_csv(input_path+"X_rf_labelled.csv")
features = list(X_temp.columns)

df = pd.merge(X_temp, df.drop(labels= df[df["MSCI_rating"].isna()].index).loc[:,["MSCI_rating", "ESG Score Grade"]], left_index=True, right_index=True)
X,y = df.loc[:,features], df.loc[:,"MSCI_rating"]
```

```python
y = simplify_msci_categories(y)
df["MSCI Category"] = y.copy()
df["ESG Category"] = simplify_categories(df["ESG Score Grade"])
```

```python
pd.crosstab(df["ESG Category"], df["MSCI Category"])
```

```python
print(f"Cramer's coeff  : {cramers_stat(df, 'ESG Category', 'MSCI Category'):.2f} | Corrected coeff : {cramers_corrected_stat(df, 'ESG Category', 'MSCI Category'):.2f}")
```

```python
categorical_countplot(
    df, 
    "kmean_labels", 
    "MSCI_rating", 
    hue_order=["aaa","aa","bbb","bb","b","ccc"],
    filename="MSCI_cat_aff.png")
```

```python
categorical_countplot(
    df, 
    "kmean_labels", 
    "MSCI Category", 
    hue_order=["a","b","c"], 
    filename="MSCI_cat_aff.png")
```

```python
params = {
    "n_estimators":100, 
    "max_depth":5,
}
```

```python
model, X_train, X_test, y_train, y_test = train_random_forest(X, y, params, test_size=0.4)
```

```python
y_pred = model.predict(X_test)
```

```python
confusion_mat_df(model, y_test, y_pred, percent=False)
```

```python
confusion_mat_df(model, y_test, y_pred, percent=True)
```

```python
pp = PrettyPrinter(indent=3)
pp.pprint(list(X.columns[-10:]))
```

```python
labels = list(X.columns[-10:])
for label in labels:
    print(f"Cramer's coeff {label} : {cramers_stat(df, label, 'MSCI_rating'):.2f} | Corrected coeff {label} : {cramers_corrected_stat(df, label, 'MSCI_rating'):.2f}")
```

```python
model, X_train, X_test, y_train, y_test = train_random_forest(
    X, 
    y, 
    params, 
    test_size=0.4, 
    with_labels=True, 
    labels="aff_labels",)
```

```python
y_pred = model.predict(X_test)
```

```python
confusion_mat_df(model, y_test, y_pred, percent=False)
```

```python
confusion_mat_df(model, y_test, y_pred, percent=True)
```

### Bibliography:
- [PCA Analysis with Python](https://towardsdatascience.com/dive-into-pca-principal-component-analysis-with-python-43ded13ead21)
- [Clustering Algorithms with Python - MachineLearningMastery](https://machinelearningmastery.com/clustering-algorithms-with-python/)
- [10 Clustering Algorithms to know - KD Nuggets](https://www.kdnuggets.com/2019/10/right-clustering-algorithm.html)
- [The Five Clustering Algorithms Dtaa Scientists need to know - Towards Data Science](https://towardsdatascience.com/the-5-clustering-algorithms-data-scientists-need-to-know-a36d136ef68)  
- [Solving the Data Dilemma in ESG Quant Investing - Invest Summit](https://www.youtube.com/watch?v=OA4axeZ-DmY)
- [Shades of Green: Investor Approaches to ESG - MSCI ](https://www.msci.com/perspectives-podcast/shades-of-green-investor-approaches-esg)
- [More Regulators Pick up the ESG Baton - KPMG](https://home.kpmg/lu/en/home/insights/2020/12/more-regulators-pick-up-the-esg-baton.html)  
- Qualitative Interviews with over 50 Sustainable Finance experts in France and Luxemburg  

- [Clustering by Passing Messages Between the datapoints](https://science.sciencemag.org/content/315/5814/972)
- Bernhard Schoelkopf, Alexander J. Smola, and Klaus-Robert Mueller. 1999. Kernel principal component analysis. In Advances in kernel methods, MIT Press, Cambridge, MA, USA 327-352.  
- [Sirus](https://hal.archives-ouvertes.fr/hal-02190689v2)

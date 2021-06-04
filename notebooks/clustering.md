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
scatterplot(prep_df, x_axis="PCA_1", y_axis="PCA_2", title="PCA scatterplot", hue=None)
```

```python
scatterplot(prep_df, x_axis="PCA_1", y_axis="PCA_2", title="PCA scatterplot by sector", hue="GICS Sector Name")
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
scatterplot(prep_df, x_axis="KPCA_1", y_axis="KPCA_2", title="KPCA scatterplot by sector", hue="GICS Sector Name")
```

```python
X_kpca = selected_kpca_features
pd.DataFrame(X_kpca).to_csv("../inputs/X_kpca.csv", index=False)
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
catplot(df, columns, figsize=(11,0.5))
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
    hue=None
)
```

```python
scatterplot(
    df, 
    x_axis="Board Gender Diversity, Percent", 
    y_axis="ESG Score", 
    title="ESG Score as a function of Board gender diversity, %",
    hue=None
)
```

```python
mask = df["Salary Gap"] < 500

scatterplot(
    df[mask], 
    x_axis="Salary Gap", 
    y_axis="ESG Score", 
    title="ESG Score as a function of Salary Gap",
    hue=None
)
```

Let's have a look at a scatterplot for an environmental indicator: 

```python
scatterplot(
    df, 
    x_axis="CO2 Equivalent Emissions Indirect, Scope 2", 
    y_axis="ESG Score", 
    title="ESG Score as a function of Scope 2 Carbon Emissions",
    hue=None
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
    hue=None
)
```

```python
scatterplot(
    X_rf, 
    x_axis="Board Gender Diversity, Percent", 
    y_axis="Market Capitalization (bil) [USD]", 
    hue="Fundamental Human Rights ILO UN", 
    title="Scatterplot"
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
#X = np.array(pd.read_csv("../inputs/X_pca.csv"))
#X = np.array(pd.read_csv("../inputs/X_kpca.csv"))
scaler = StandardScaler()
X = scaler.fit_transform(np.array(pd.read_csv("../inputs/X_rf.csv")))
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
sse = kmeans(X, kmeans_kwargs, upper=50, plot=True)
```

```python
optimal_nb = 10
```

```python
optimal_kmeans = KMeans(n_clusters=optimal_nb, **kmeans_kwargs)
optimal_kmeans.fit(X)
```

```python
prep_df["kmean_labels"] = optimal_kmeans.labels_
#X["kmean_labels"] = optimal_kmeans.labels_
```

```python
prep_df["kmean_labels"].value_counts()
```

```python
prep_df["ESG Category"] = simplify_categories(prep_df["ESG Score Grade"])
```

```python
plt.figure(figsize=(17,8))
sns.countplot(
    x="kmean_labels", 
    hue="ESG Category", 
  #  order = ["A", "B", "C", "D"],
    data=prep_df, 
    palette="husl", 
)
plt.title("ESG Category repartition by cluster.")
plt.show()
```

To run our data exploration, we perform it on the initial dataframe. 

```python
df = pd.read_csv("../inputs/universe_df_no_nans.csv").drop(columns=["Unnamed: 0"])
df.loc[:,"kmean_labels"] = X_rf.loc[:,"kmean_labels"]
```

#### Mini-batch Kmeans

```python
sse = m_kmeans(X, upper=50, plot=True)
```

```python
optimal_nb = 10
```

```python
optimal_mkmeans = MiniBatchKMeans(n_clusters=optimal_nb)
optimal_mkmeans.fit(X)
yhat = optimal_mkmeans.predict(X)
```

```python
prep_df["mkmean_labels"] = optimal_mkmeans.labels_
#X_rf["mkmean_labels"] = optimal_mkmeans.labels_
```

```python
prep_df["mkmean_labels"].value_counts()
```

```python
plt.figure(figsize=(17,8))
sns.countplot(
    x="mkmean_labels", 
    hue="ESG Category", 
  #  order = ["A", "B", "C", "D"],
    data=prep_df, 
    palette="husl", 
)
plt.title("ESG Category repartition by cluster.")
plt.show()
```

```python
scatterplot(
    prep_df, 
    "KPCA_1", 
    "KPCA_2", 
    "mkmean_labels", 
    "KPCA coloured by Mini-kmeans"
)
```

#### Affinity propagation


Affinity propagation finds "exemplars," members of the input set that are representative of clusters.

```python
damping = 0.5
```

```python
# define the model
aff_model = AffinityPropagation(damping=damping)
# fit the model
aff_model.fit(X)
# assign a cluster to each example
yhat = aff_model.predict(X)
```

```python
prep_df["aff_labels"] = aff_model.labels_
X_rf["aff_labels"] = aff_model.labels_
```

```python
X_rf["aff_labels"].value_counts()
```

```python
scatterplot(
    prep_df, 
    "KPCA_1", 
    "KPCA_2", 
    "aff_labels", 
    "KPCA coloured by Affinity propagation"
)
```

#### Agglomerative clustering


Belongs to hierarchical clustering methods.

```python
# define the model
agg_model = AgglomerativeClustering(n_clusters=optimal_nb)
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
    "KPCA coloured by Aggregation"
)
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
birch_model.fit(X)
# assign a cluster to each example
yhat = birch_model.predict(X)
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
    "KPCA coloured by Mini-kmeans"
)
```

#### DBSCAN clustering


Used to find clusters of arbitrary shape.

```python
eps = 0.30
min_samples =10
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
scatterplot(
    prep_df, 
    "KPCA_1", 
    "KPCA_2", 
    "dbscan_labels", 
    "KPCA coloured by Mini-kmeans"
)
```

#### Mean Shift clustering


Mean shift clustering involves finding and adapting centroids based on the density of examples in the feature space.

```python
# define the model
mshift_model = MeanShift(min_bin_freq=100)
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
    "KPCA coloured by Mean-shift"
)
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
    "KPCA coloured by Optics"
)
```

#### Spectral clustering

```python
optimal_nb = 20
```

Here, one uses the top eigenvectors of a matrix derived from the distance between points.

```python
# define the model
spec_model = SpectralClustering(n_clusters=optimal_nb)
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
    "KPCA coloured by Spectral labels"
)
```

#### Gaussian Mixture Clustering


A Gaussian mixture model summarizes a multivariate probability density function with a mixture of Gaussian probability distributions as its name suggests.

```python
optimal_nb = 15
```

```python
# define the model
gmm_model = GaussianMixture(n_components=optimal_nb)
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
    "KPCA coloured by Gaussian Labels"
)
```

```python
#X_rf.to_csv("../inputs/X_rf_labelled.csv", index=False)
#prep_df.to_csv("../inputs/prep_df_labelled.csv", index=False)
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
df["ESG Category"].value_counts().index
```

```python
plt.figure(figsize=(17,8))
sns.countplot(
    x="kmean_labels", 
    hue="ESG Category", 
  #  order = ["A", "B", "C", "D"],
    data=df, 
    palette="husl", 
)
plt.title("ESG Category repartition by cluster.")
plt.show()
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
df = pd.DataFrame(X,y)
```

```python
y = simplify_categories(y)
```

```python
params = {
    "n_estimators":500, 
    "max_depth":10,
}
```

```python
model, X_train, X_test, y_train, y_test = train_random_forest(X, y, params)
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
model, X_train, X_test, y_train, y_test = train_random_forest(
    X, 
    y, 
    params, 
    test_size=0.4, 
    with_labels=True, 
    labels="kmean_labels",)
```

```python
confusion_mat_df(model, y_test, y_pred, percent=False)
```

```python
confusion_mat_df(model, y_test, y_pred, percent=True)
```

```python
countplot(pred_df,"ESG Score Grade", filename="ESG_score_distribution.png")
sns.countplot(x="MSCI_rating", hue="GICS Sector Name", data=pred_df)
plt.figsize((10,10))
plt.show()
countplot(pred_df,"MSCI_rating", filename="ESG_score_distribution.png")
columns = ["GICS Sector Name", "ESG Score"]
boxplot(pred_df, columns, filename="ESG_score_sector.png", categorical=True)
columns = ["GICS Sector Name", "MSCI_rating"]
boxplot(pred_df, columns, filename="MSCI_ESG_score_sector.png", categorical=True)
#scatterplot(prep_df, x_axis="PCA_1", y_axis="PCA_2", title="PCA scatterplot by sector", hue="ESG Score Grade")
#scatterplot(prep_df, x_axis="PCA_1", y_axis="PCA_2", title="PCA scatterplot by sector", hue="Environmental Pillar Score Grade")
#scatterplot(prep_df, x_axis="PCA_1", y_axis="PCA_2", title="PCA scatterplot by sector", hue="Social Pillar Score Grade")
#scatterplot(prep_df, x_axis="PCA_1", y_axis="PCA_2", title="PCA scatterplot by sector", hue="Governance Pillar Score Grade")

scatterplot(
    prep_df, 
    x_axis="PCA_1", 
    y_axis="PCA_2", 
    title="PCA scatterplot by ESG Category", 
    hue="ESG Categories"
)

scatterplot(
    prep_df, 
    x_axis="PCA_1", 
    y_axis="PCA_2", 
    title="PCA scatterplot by Environmental Category", 
    hue="Environmental Categories"
)

scatterplot(
    prep_df, 
    x_axis="PCA_1", 
    y_axis="PCA_2", 
    title="PCA scatterplot by Social Category", 
    hue="Social Categories"
)
scatterplot(
    prep_df, 
    x_axis="PCA_1", 
    y_axis="PCA_2", 
    title="PCA scatterplot by Governance Category", 
    hue="Governance Categories"
)
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

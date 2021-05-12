<!-- #region -->
# Data exploration and initial clustering.
In this notebook, our reference universe will correspond to all authorized stocks within the **Natixis Investment Managers Challenge, 2021 edition**. It comprises over **2,000 companies** for which we retrieved ESG data for year 2020. 
  

 
Our goal is to use **Sustainable Finance Disclosure Regulation (SFDR)** metrics to perform clustering on these companies without using Refinitiv's own notation. Ultimately, we would like to build our own explainable score for the data.
<!-- #endregion -->

First, run all necessary imports.

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pprint
from sklearn.preprocessing import OneHotEncoder
```

```python
# Utils

## Data visualisation
def boxplot(df, columns, categorical=True, figsize=(10,8)):
    """
    Draw categorical boxplot for the series of your choice, excluding missing values. 
    """
    data = df.loc[:,columns].dropna()
    print(f"Boxplot for {len(data)} datapoints out of {len(df)} overall.")
    
    fig, ax = plt.subplots(figsize=figsize)
    plt.setp(ax.get_xticklabels(), rotation=45)
    if categorical == True:
        sns.boxplot(
            x=columns[0],
            y=columns[1], 
            data=data
        )
    else: 
        sns.boxplot(
            y=columns[1], 
            data=data
        )
        
def countplot(df, category, figsize=(10,6)):
    """ 
    Countplot for the category of your choice. 
    """
    fig, ax = plt.subplots(figsize=figsize)
    plt.setp(ax.get_xticklabels(), rotation=45)
    sns.countplot(
        x=category, 
        data=df,
        order = df[category].value_counts().index)
    plt.title(f"Countplot by {category}")
    plt.show()
    

## Data preprocessing
def fillna(df, sectors, median_strategy_cols, conservative_cols, drop_cols):
    """
    Fill missing values for our dataframe. 
    """
    df.loc[:,"Critical Country 1":"Critical Country 4"] = df.loc[:,"Critical Country 1":"Critical Country 4"].fillna("None")
    
    # Fix GICs industry
    df.iloc[306,7] = "Electronical Equipment"
    df.iloc[1739,7] = "Biotechnology"
    
    
    # Fix NaNs by industry
    for sector in sectors:
        mask = df.loc[:,"GICS Sector Name"]==sector
        # Median strategy
        for feature in median_strategy_cols:
            nan_value = df[mask].loc[:,feature].median()
            rows = list(df[mask].index)
            col = df.columns.get_loc(feature)
            df.iloc[rows, col] = df.iloc[rows, col].fillna(nan_value)
        
        # Conservative hypothesis
        for category in conservative_cols:
            nan_value = 0
            rows = list(df[mask].index)
            col = df.columns.get_loc(category)
            df.iloc[rows, col] = df.iloc[rows, col].fillna(nan_value)
            
    
    df = df.drop(columns=drop_cols)
    return df.dropna()

def one_hot_encode(df, categorical_cols):
    """
    One Hot Encode our data. 
    """
    enc = OneHotEncoder()
    X = df.loc[:,categorical_cols]
    enc.fit(X)
    temp = pd.DataFrame(enc.transform(X).toarray())
    temp.columns = enc.get_feature_names()
    out = pd.concat([df.reset_index(),temp.reset_index()], axis=1).drop(columns=["index"])
    return out

```

Then, download your data.

```python
input_path = "../inputs/"
input_filename = "universe_df_full_scores.csv"
output_path = "../outputs/"
```

```python
initial_df = pd.read_csv(input_path+input_filename)
```

```python
initial_df = initial_df.drop_duplicates(subset=['ISINS']).reset_index().drop(columns=["index"])
initial_df = initial_df.drop_duplicates(subset=['Name']).reset_index().drop(columns=["Unnamed: 0","Unnamed: 0.1", "index"])
initial_df = initial_df.drop(columns=["CARBON_FOOTPRINT"])
```

```python
initial_df.head()
```

Here you will find the list of all SFDR metrics we subselected to make our choice:

```python
sfdr_metrics = {
    'TR.GICSSector': 'GIC_Sector',
    'TR.NACEClassification': 'NACE_Sector',
    'TR.CO2EmissionTotal': "GHG Emissions",
    'TR.CO2DirectScope1': "GHG Emissions",
    'TR.CO2IndirectScope2': "GHG Emissions",
    'TR.CO2IndirectScope3': "GHG Emissions",
    'carbon_footprint': "GHG Emissions",
    'TR.AnalyticCO2': "GHG Emissions",
   # 'TR.EnergyUseTotal':"Energy Efficiency",
    'TR.AnalyticTotalRenewableEnergy':"Energy Efficiency", # il faut faire 1-ça
    'TR.AnalyticEnergyUse':'Energy Efficiency', # globally and by NACE sector, GJ/M$
    'TR.BiodiversityImpactReduction':"Biodiversity", # does the company monitor its impact
    'TR.AnalyticDischargeWaterSystem':"Water", # ton emissions / $M
    'TR.HazardousWaste': "Waste",
    'TR.WasteTotal':'Waste', # to get non recycled waste
    'TR.WasteRecycledTotal':'Waste', 
    'TR.ILOFundamentalHumanRights': 'Social and Employee Matters',
    'TR.GenderPayGapPercentage':'Social and Employee Matters', # women to men
    'TR.AnalyticSalaryGap':'Social and Employee Matters', # to average, should be median
    'TR.AnalyticBoardFemale': 'Social and Employee Matters', 
    'TR.WhistleBlowerProtection': 'Social and Employee Matters',
    'TR.AccidentsTotal': 'Social and Employee Matters', # proxy for accidents
    'TR.AnalyticHumanRightsPolicy': 'Social and Employee Matters',
    'TR.CriticalCountry1': 'Social and Employee Matters', # as a proxy for operations at risk of child or forced labour
    'TR.CriticalCountry2': 'Social and Employee Matters', # as a proxy for operations at risk of child or forced labour
    'TR.CriticalCountry3': 'Social and Employee Matters', # as a proxy for operations at risk of child or forced labour
    'TR.CriticalCountry4': 'Social and Employee Matters', # as a proxy for operations at risk of child or forced labour
    'TR.AntiPersonnelLandmines':'Social and Employee Matters', # anti personnel landmines
    'TR.PolicyBriberyCorruption': 'Anti-corruption and Anti-bribery',
    'TR.AnalyticBriberyFraudControv':'Anti-corruption and Anti-bribery',
}
```

### Data exploration without using any ESG scoring.


We exclude ESG notations by Refinitiv from our data exploration, as we will use them only later for verification purposes.  
Note that Refinitiv adopts a **best-in-class** methodology, meaning that our clusters may not correspond to theirs as we run our analysis for all fields together. 

```python
# Simply remove ESG notations for our analysis
df = initial_df.loc[:,"Name":"Bribery, Corruption and Fraud Controversies"].copy()
```

```python
pp = pprint.PrettyPrinter(indent=4)
pp.pprint(df.columns)
```

```python
print(f"Our dataframe presents {df.shape[1]} SFDR-related metrics.")
```

Let's observe the geographic repartition of our dataset.

```python
countplot(df,"Country")
```

The USA is over-represned in our dataset. We only have Western companies.

```python
countplot(df,"GICS Sector Name", figsize=(8,8))
```

```python
countplot(df,"Whistleblower Protection", (6,6))
```

Energy, Materials and Utilities present a higher average CO2 emissions total than other industries, with quite a few outliers towards the higher extreme.

```python
columns = ["GICS Sector Name","CO2 Equivalent Emissions Total"]
boxplot(df, columns, categorical=True, figsize=(10,6))
```

```python
columns = ["GICS Sector Name","Board Gender Diversity, Percent"]
boxplot(df, columns, categorical=False, figsize=(6,6))
```

```python
columns = ["GICS Sector Name","Board Gender Diversity, Percent"]
boxplot(df, columns, categorical=True)
```

```python
columns = ["GICS Sector Name","CO2 Equivalent Emissions Indirect, Scope 2"]
boxplot(df, columns, categorical=True)
```

```python
columns = ["GICS Sector Name","CO2 Equivalent Emissions Indirect, Scope 3"]
boxplot(df, columns, categorical=True)
```

```python
columns = ["GICS Sector Name","CO2 Equivalent Emissions Total"]
boxplot(df, columns, categorical=True)
```

### Data preprocessing

```python
# Add columns to full df
# Letters
#prep_df["ESG Score Grade"] = initial_df.loc[:,"ESG Score Grade"]
#prep_df['Environmental Pillar Score Grade'] = initial_df.loc[:,"Environmental Pillar Score Grade"]
#prep_df['Social Pillar Score Grade'] = initial_df.loc[:,"Social Pillar Score Grade"]
#prep_df['Governance Pillar Score Grade'] = initial_df.loc[:,"Governance Pillar Score Grade"]
```

```python
df["ESG Score"] = initial_df["ESG Score"]
df['Environmental Pillar Score'] = initial_df.loc[:,"Environmental Pillar Score"]
df['Social Pillar Score'] = initial_df.loc[:,"Social Pillar Score"]
df['Governance Pillar Score'] = initial_df.loc[:,"Governance Pillar Score"]
```

```python
(df.isna().sum().sort_values()/len(df))*100
```

```python
df.shape
```

We first note that some metrics, although they are required by SFDR, suffer from particularly low coverage! This confirms observations by industry experts throughout our qualitative interviews.  
For instance, carbon emissions total stands at approximately 50% missing datapoints, and water pollutant emissions to revenues information is practically unavaiable at 99% missing datapoints.


Depending on the missing value strategy you want to put in place by column, change column names here:

```python
sectors = list(df["GICS Sector Name"].value_counts().index)

# Set value to median sector value. 
median_strategy_cols = [
    "Board Gender Diversity, Percent",
    "CO2 Equivalent Emissions Total",
    "Total CO2 Equivalent Emissions To Revenues USD in million",
    "CO2 Equivalent Emissions Direct, Scope 1",
    "CO2 Equivalent Emissions Indirect, Scope 2",
    "Total Energy Use To Revenues USD in million",
    "Salary Gap",
    "Waste Total",
    "CO2 Equivalent Emissions Indirect, Scope 3",
    "Waste Recycled Total",
    "Accidents Total",
    "Total Renewable Energy To Energy Use in million",
    "Hazardous Waste",
    "ESG Score",
    "Environmenal Pillar Score",
    "Social Pillar Score",
    "Governance Pillar Score"
]

# Assuming there is no policy in place, value set to zero. 
conservative_cols = [
    "Anti-Personnel Landmines",
    "Fundamental Human Rights ILO UN",
    "Biodiversity Impact Reduction",
    "Human Rights Policy",
    "Whistleblower Protection",
    "Bribery, Corruption and Fraud Controversies",
]

# Drop these columns. 
drop_cols = [
    "Water Pollutant Emissions To Revenues USD in million",
    "NACE Classification"
]
```

```python
df_fillna = fillna(df, sectors, median_strategy_cols, conservative_cols, drop_cols)
```

```python
df_fillna.head()
```

```python
df_fillna.shape
```

Then, define the columns to encode using a OneHotEncoder. 

```python
categorical_cols = [
   # "Instrument",to keep trace
    "Country",
    "GICS Sector Name",
    "Industry Name - GICS Sub-Industry",
   # "NACE Classification",
   # "Biodiversity Impact Reduction",
   # "Fundamental Human Rights ILO UN",
   # "Whistleblower Protection",
   # "Human Rights Policy",
    "Critical Country 1",
    "Critical Country 2",
    "Critical Country 3",
    "Critical Country 4",
   # "Anti-Personnel Landmines",
   # "Bribery, Corruption and Fraud Controversies"
]
```

```python
df_encoded = one_hot_encode(df_fillna, categorical_cols)
```

```python
df_encoded.head()
```

```python
df_encoded.to_csv("../inputs/universe_df_encoded.csv")
```

### Basic clustering

```python
import numpy as np
from matplotlib.pyplot import figure
from numpy import unique
from numpy import where
from matplotlib import pyplot
from sklearn.preprocessing import StandardScaler

#Importing required modules
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from sklearn.ensemble import RandomForestRegressor
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

```python
# Utils 

def pca_selection(cluster_df, n_components=200, threshold=0.5, method="cumsum"):
    """
    PCA algorithm for feature selection. 
    """
    features = np.array(cluster_df)
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    pca = PCA(n_components)
    pca_features = pca.fit_transform(scaled_features)
    
    if method == "cumsum":
        cumsum = np.cumsum(pca.explained_variance_ratio_)
        last_feature = np.where(cumsum > threshold)[0][0]-1
        return pca_features, pca, pca_features[:,:last_feature]
        
    if method == "feature-wise":
        mask = pca.explained_variance_ratio_ > threshold
        return pca_features, pca, pca_features[:,np.where(mask == True)[0]]
    

def kernel_pca_selection(cluster_df, n_components=200, threshold=0.5, method="cumsum"):
    """
    PCA algorithm for feature selection. 
    """
    features = np.array(cluster_df)
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    kernel_pca = KernelPCA(n_components)
    kernel_pca_features = kernel_pca.fit_transform(scaled_features)
    
    if method == "cumsum":
        cumsum = np.cumsum(pca.explained_variance_ratio_)
        last_feature = np.where(cumsum > threshold)[0][0]-1
        return pca_features, pca, pca_features[:,:last_feature]
        
    if method == "feature-wise":
        mask = pca.explained_variance_ratio_ > threshold
        return pca_features, pca, pca_features[:,np.where(mask == True)[0]]
    

def random_forest_selection(X,y, threshold=0.3):
    """
    Feature selection by using a random forest regressor.
    """
    rf_reg = RandomForestRegressor(n_estimators=100)
    rf_reg.fit(X,y)
    
    importances = pd.DataFrame(list(cluster_df.columns))
    importances["feature_importances"] = rf_reg.feature_importances_
    importances.columns = ["features", "importances"]
    importances = importances.sort_values(by=["importances"],ascending=False).reset_index().copy()
    
    #importances[:n_features].loc[:,"features":"importances"].plot(kind="barh")
    mask = importances["importances"] > threshold
    n_features = len(importances[mask])
    importances[mask].loc[:,"features":"importances"].plot(kind="barh")
    plt.yticks(ticks=range(n_features),labels=importances[:n_features]["features"])
    plt.title(f"Top {n_features} feature importance for threshold of {threshold}")
    plt.show()
    
    return importances
```

```python
prep_df = pd.read_csv("../inputs/universe_df_encoded.csv")
```

```python
prep_df.head()
```

### Dimensionality reduction techniques


#### a) PCA


Note that PCA works for data that is linearly separable. Given the complexity of our data, this may well not be the case as we will explore here.


First, we drop non-numerical columns.

```python
drop_cols = [
    "Unnamed: 0",
    "Name",
    "Symbol",
    "Country",
    "Industry Name - GICS Sub-Industry",
    "SEDOL",
    "ISINS",
    "GICS Sector Name",
    "Critical Country 1",
    "Critical Country 2",
    "Critical Country 3",
    "Critical Country 4",
    "ESG Score"
]
cluster_df = prep_df.drop(columns=drop_cols).copy()
```

```python
cluster_df.columns
```

```python
cluster_df.shape
```

We then scale our data to be able to perform a PCA on it using scikit learn.

```python
pca_features, pca, X = pca_selection(cluster_df, threshold=0.8, method="cumsum")
```

```python
ticks = list(range(n_components))[::10]
labels = [x+1 for x in ticks]

figure(figsize=(10,4))
plt.plot(pca.explained_variance_,)
plt.xlabel("Number of components")
plt.ylabel("Explained variance")
plt.xticks(ticks=ticks, labels=labels)
plt.title("PCA feature selection")
plt.show()
```

```python
figure(figsize=(10,4))
plt.plot(pca.explained_variance_ratio_)
plt.xlabel("Number of components")
plt.ylabel("Explained variance %")
plt.xticks(ticks=ticks, labels=labels)
plt.title("PCA feature selection")
plt.show()
```

```python
prep_df["PCA_1"] = pd.Series(pca_features[:,0])
prep_df["PCA_2"] = pd.Series(pca_features[:,1])
```

### Kernel PCA 
Applying a kernel to make the data linearly separable by PCA. 

```python
X = np.array(cluster_df)
n_components = 2
```

```python
kernel_pca = KernelPCA(n_components=n_components, kernel='rbf')
kernel_pca_features = kernel_pca.fit_transform(X)
```

```python
kernel_pca.lambdas_
np.cov(kernel_pca_features.T)
```

```python
plt.scatter(x=kernel_pca_features[:,0], y=kernel_pca_features[:,1])
plt.title("Kernel PCA feature scatterplot")
plt.show()
```

### Random forest


We train a Random Forest to predict the ESG Score as a function of our SFDR metrics, and use a threshold feature importance to select relevant features.  

```python
threshold = 0.005
```

```python
X, y = cluster_df.copy(), prep_df["ESG Score"].copy()
importances = random_forest_selection(X,y, threshold=threshold)
```

```python
mask = importances.importances > threshold
X = cluster_df.loc[:,importances[mask].features]
X.head()
```

### Autoencoder

```python
# to do 
```

We remove them from the dataframe.

```python
prep_df = prep_df.drop(index=drop_rows).copy()
```

```python
pca_features = np.array(prep_df.loc[:,["PCA_1","PCA_2"]])
pca_features
```

```python
plt.scatter(prep_df.loc[:,"PCA_1"], prep_df.loc[:,"PCA_2"])
plt.title("Data plot after 2-dimensional PCA.")
plt.show()
```

```python
sns.scatterplot(data = prep_df, x = prep_df.loc[:,"PCA_1"] , y = prep_df.loc[:,"PCA_2"] , hue = 'GICS Sector Name')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.title("PCA plot coloured by industry.")
plt.show()
```

```python
sns.scatterplot(data = prep_df, x = prep_df.loc[:,"PCA_1"] , y = prep_df.loc[:,"PCA_2"] , hue = 'ESG Score Grade')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.title("PCA plot coloured by ESG Score Grade.")
plt.show()
```

```python
# PCA plot coloured by overall E score
sns.scatterplot(data = prep_df, x = prep_df.loc[:,"PCA_1"] , y = prep_df.loc[:,"PCA_2"] , hue = 'Environmental Pillar Score Grade')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.title("PCA plot coloured by Environmental Pillar Score Grade.")
plt.show()
```

```python
# PCA plot coloured by overall S score
sns.scatterplot(data = prep_df, x = prep_df.loc[:,"PCA_1"] , y = prep_df.loc[:,"PCA_2"] , hue = 'Social Pillar Score Grade')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.title("PCA plot coloured by Social Pillar Score Grade.")
plt.show()
```

```python
# PCA plot coloured by overall G score
sns.scatterplot(data = prep_df, x = prep_df.loc[:,"PCA_1"] , y = prep_df.loc[:,"PCA_2"] , hue = 'Governance Pillar Score Grade')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.title("PCA plot coloured by Governance Pillar Score Grade.")
plt.show()
```

### Clustering


As such, cluster analysis is an iterative process where subjective evaluation of the identified clusters is fed back into changes to algorithm configuration until a desired or appropriate result is achieved.


We identify some outliers along the x-axis. Let's performe the elbow method to determine the optimal number of clusters.


#### K-means


Assign examples to each cluster while trying to minimize the variance within each cluster.

```python
kmeans_kwargs = {
    "init": "random",
    "n_init": 10,
    "max_iter": 300,
    "random_state": 42,
}

lower, upper = 1, 10
# A list holds the SSE values for each k
sse = []
for k in range(lower, upper):
    kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
    kmeans.fit(pca_features)
    sse.append(kmeans.inertia_)
```

```python
plt.style.use("fivethirtyeight")
plt.plot(range(lower+1, upper+1), sse)
plt.xticks(range(lower+1, upper+1))
plt.xlabel("Number of Clusters")
plt.ylabel("SSE")
plt.title("Elbow graph for KMeans")
plt.show()
```

Based on this elbow graph, we select 4 as the optimal number of clusters.

```python
optimal_nb = 6
```

```python
optimal_kmeans = KMeans(n_clusters=optimal_nb, **kmeans_kwargs)
optimal_kmeans.fit(pca_features)
```

```python
prep_df["kmean_labels"] = optimal_kmeans.labels_
prep_df.head()
```

Now, let's plot our datapoints for each kmeans label.

```python
sns.scatterplot(data = prep_df, x = prep_df.loc[:,"PCA_1"] , y = prep_df.loc[:,"PCA_2"] , hue = 'kmean_labels')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.title("PCA plot coloured by kmeans.")
plt.show()
```

```python
X = pca_features
```

#### Mini-batch Kmeans

```python
# define the model
sse = []
for k in range(lower, upper):
    model = MiniBatchKMeans(n_clusters=k)
    model.fit(X)
    sse.append(model.inertia_)
```

```python
plt.style.use("fivethirtyeight")
plt.plot(range(lower+1, upper+1), sse)
plt.xticks(range(lower+1, upper+1))
plt.xlabel("Number of Clusters")
plt.ylabel("SSE")
plt.title("Elbow graph for Mini-batch KMeans")
plt.show()
```

```python
optimal_nb = 6
```

```python
model = MiniBatchKMeans(n_clusters=optimal_nb)
# fit the model
model.fit(X)
# assign a cluster to each example
yhat = model.predict(X)
# retrieve unique clusters
clusters = unique(yhat)
# create scatter plot for samples from each cluster
for cluster in clusters:
    # get row indexes for samples with this cluster
    row_ix = where(yhat == cluster)
    # create scatter of these samples
    plt.scatter(X[row_ix, 0], X[row_ix, 1])
# show the plot
plt.title("Mini-batch Kmeans")
plt.show()
```

#### Affinity propagation


Affinity propagation finds "exemplars," members of the input set that are representative of clusters.

```python
# define the model
model = AffinityPropagation(damping=0.5)
# fit the model
model.fit(X)
# assign a cluster to each example
yhat = model.predict(X)
# retrieve unique clusters
clusters = unique(yhat)
# create scatter plot for samples from each cluster
for cluster in clusters:
    # get row indexes for samples with this cluster
    row_ix = where(yhat == cluster)
    # create scatter of these samples
    plt.scatter(X[row_ix, 0], X[row_ix, 1])
# show the plot
plt.title("Affinity Propagation")
plt.show()
```

#### Agglomerative clustering


Belongs to hierarchical clustering methods.

```python
# define the model
model = AgglomerativeClustering(n_clusters=optimal_nb)
# fit model and predict clusters
yhat = model.fit_predict(X)
# retrieve unique clusters
clusters = unique(yhat)
# create scatter plot for samples from each cluster
for cluster in clusters:
    # get row indexes for samples with this cluster
    row_ix = where(yhat == cluster)
    # create scatter of these samples
    plt.scatter(X[row_ix, 0], X[row_ix, 1])
# show the plot
plt.title("Affinity Propagation")
plt.show()
```

#### BIRCH Clustering


Constructing a tree structure from which cluster centroids are extracted.

```python
threshold = 0.01
n_clusters = optimal_nb
```

```python
# define the model
model = Birch(threshold=threshold, n_clusters=n_clusters)
# fit the model
model.fit(X)
# assign a cluster to each example
yhat = model.predict(X)
# retrieve unique clusters
clusters = unique(yhat)
# create scatter plot for samples from each cluster
for cluster in clusters:
    # get row indexes for samples with this cluster
    row_ix = where(yhat == cluster)
    # create scatter of these samples
    plt.scatter(X[row_ix, 0], X[row_ix, 1])
# show the plot
plt.title("BIRCH")
plt.show()
```

#### DBSCAN clustering


Used to find clusters of arbitrary shape.

```python
eps = 0.30
min_samples = 9
```

```python
# define the model
model = DBSCAN(eps=eps, min_samples=min_samples)
# fit model and predict clusters
yhat = model.fit_predict(X)
# retrieve unique clusters
clusters = unique(yhat)
# create scatter plot for samples from each cluster
for cluster in clusters:
    # get row indexes for samples with this cluster
    row_ix = where(yhat == cluster)
    # create scatter of these samples
    plt.scatter(X[row_ix, 0], X[row_ix, 1])
# show the plot
plt.title("DBSCAN")
plt.show()
```

#### Mean Shift clustering


Mean shift clustering involves finding and adapting centroids based on the density of examples in the feature space.

```python
# define the model
model = MeanShift()
# fit model and predict clusters
yhat = model.fit_predict(X)
# retrieve unique clusters
clusters = unique(yhat)
# create scatter plot for samples from each cluster
for cluster in clusters:
    # get row indexes for samples with this cluster
    row_ix = where(yhat == cluster)
    # create scatter of these samples
    plt.scatter(X[row_ix, 0], X[row_ix, 1])
# show the plot
plt.title("Mean Shift")
plt.show()
```

### OPTICS


Modified version of DBSCAN. 

```python
model = OPTICS(eps=0.8, min_samples=10)
# fit model and predict clusters
yhat = model.fit_predict(X)
# retrieve unique clusters
clusters = unique(yhat)
# create scatter plot for samples from each cluster
for cluster in clusters:
	# get row indexes for samples with this cluster
	row_ix = where(yhat == cluster)
	# create scatter of these samples
	plt.scatter(X[row_ix, 0], X[row_ix, 1])
# show the plot
plt.title("OPTICS")
plt.show()
```

#### Spectral clustering


Here, one uses the top eigenvectors of a matrix derived from the distance between points.

```python
# define the model
model = SpectralClustering(n_clusters=optimal_nb)
# fit model and predict clusters
yhat = model.fit_predict(X)
# retrieve unique clusters
clusters = unique(yhat)
# create scatter plot for samples from each cluster
for cluster in clusters:
    # get row indexes for samples with this cluster
    row_ix = where(yhat == cluster)
    # create scatter of these samples
    plt.scatter(X[row_ix, 0], X[row_ix, 1])
# show the plot
plt.title("Spectral Clustering")
plt.show()
```

#### Gaussian Mixture Clustering


A Gaussian mixture model summarizes a multivariate probability density function with a mixture of Gaussian probability distributions as its name suggests.

```python
# define the model
model = GaussianMixture(n_components=optimal_nb)
# fit the model
model.fit(X)
# assign a cluster to each example
yhat = model.predict(X)
# retrieve unique clusters
clusters = unique(yhat)
# create scatter plot for samples from each cluster
for cluster in clusters:
    # get row indexes for samples with this cluster
    row_ix = where(yhat == cluster)
    # create scatter of these samples
    plt.scatter(X[row_ix, 0], X[row_ix, 1])
# show the plot
plt.title("Gaussian Mixture")
plt.show()
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




## Secondary analysis to identify what affects ESG score in Refinitiv Data.

```python
grades_dict = {
    'A+':  'A', 
    'A':  'A', 
    'A-':  'A',
    'B+': 'B', 
    'B-': 'B', 
    'B':  'B', 
    'C+': 'C', 
    'C':  'C', 
    'C-':  'C', 
    'D+':  'D', 
    'D':  'D', 
    'D-':  'D',
}
```

```python
prep_df["ESG categories"] = prep_df["ESG Score Grade"].map(grades_dict)
prep_df["Environmental categories"] = prep_df["Environmental Pillar Score Grade"].map(grades_dict)
prep_df["Social categories"] = prep_df["Social Pillar Score Grade"].map(grades_dict)
prep_df["Governance categories"] = prep_df["Governance Pillar Score Grade"].map(grades_dict)
```

```python
sns.scatterplot(data = prep_df, x = prep_df.loc[:,"PCA_1"] , y = prep_df.loc[:,"PCA_2"] , hue = 'ESG categories')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.title("PCA plot coloured by ESG  Category.")
plt.show()
```

```python
sns.scatterplot(data = prep_df, x = prep_df.loc[:,"PCA_1"] , y = prep_df.loc[:,"PCA_2"] , hue = 'Environmental categories')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.title("PCA plot coloured by Environmental Category.")
plt.show()
```

```python
sns.scatterplot(data = prep_df, x = prep_df.loc[:,"PCA_1"] , y = prep_df.loc[:,"PCA_2"] , hue = 'Social categories')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.title("PCA plot coloured by Social Category.")
plt.show()
```

```python
sns.scatterplot(data = prep_df, x = prep_df.loc[:,"PCA_1"] , y = prep_df.loc[:,"PCA_2"] , hue = 'Governance categories')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.title("PCA plot coloured by Governance Category.")
plt.show()
```

```python
fig,axes =plt.subplots(10,3, figsize=(12, 9)) # 3 columns each containing 10 figures, total 30 features
excellent= prep_df.loc[prep_df["ESG categories"]=="A"] # define malignant
bad=prep_df.loc[prep_df["ESG categories"]=="D"] # define benign
ax=axes.ravel()# flat axes with numpy ravel
for i in range(30):
  _,bins=np.histogram(prep_df.iloc[:,i],bins=40)
  ax[i].hist(excellent.iloc[:,i],bins=bins,color='r',alpha=.5)# red color for malignant class
  ax[i].hist(bad.iloc[:,i],bins=bins,color='g',alpha=0.3)# alpha is           for transparency in the overlapped region 
  ax[i].set_title(prep_df.columns[i],fontsize=9)
  ax[i].axes.get_xaxis().set_visible(False) # the x-axis co-ordinates are not so useful, as we just want to look how well separated the histograms are
  ax[i].set_yticks(())
ax[0].legend(['excellent','bad'],loc='best',fontsize=8)
plt.tight_layout()# let's make good plots
plt.show()
```

```python

```

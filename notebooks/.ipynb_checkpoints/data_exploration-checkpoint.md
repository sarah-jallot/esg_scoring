First, run all necessary imports.

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
```

```python
# Utils

## Data visualisation
def boxplot(df, columns, categorical=True):
    """
    Draw categorical boxplot for the series of your choice, excluding missing values. 
    """
    data = df.loc[:,columns].dropna()
    print(f"Boxplot for {len(data)} datapoints out of {len(df)} overall.")
    
    fig, ax = plt.subplots(figsize=(10,8))
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
        
def countplot(df, category):
    """ 
    Countplot for the category of your choice. 
    """
    fig, ax = plt.subplots(figsize=(10,6))
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

```python
df = initial_df.loc[:,"Name":"Bribery, Corruption and Fraud Controversies"].copy()
```

```python
df.columns
```

```python
countplot(df,"Country")
```

```python
countplot(df,"GICS Sector Name")
```

Energy, Materials and Utilities present a higher average CO2 emissions total than other industries, with quite a few outliers towards the higher extreme.

```python
columns = ["GICS Sector Name","CO2 Equivalent Emissions Total"]
boxplot(df, columns, categorical=True)
```

```python
columns = ["GICS Sector Name","Board Gender Diversity, Percent"]
boxplot(df, columns, categorical=False)
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
(df.isna().sum().sort_values()/len(df))*100
```

Depending on the missing value strategy you want to put in place by column, change column names here:

```python
sectors = list(df["GICS Sector Name"].value_counts().index)
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
]

conservative_cols = [
    "Anti-Personnel Landmines",
    "Fundamental Human Rights ILO UN",
    "Biodiversity Impact Reduction",
    "Human Rights Policy",
    "Whistleblower Protection",
    "Bribery, Corruption and Fraud Controversies",
]

drop_cols = [
    "Water Pollutant Emissions To Revenues USD in million",
    "NACE Classification"
]
```

```python
df_fillna = fillna(df, sectors, median_strategy_cols, conservative_cols, drop_cols)
```

```python
df_fillna
```

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

from numpy import unique
from numpy import where
from matplotlib import pyplot
from sklearn.preprocessing import StandardScaler

#Importing required modules
from sklearn.decomposition import PCA
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
```

```python
prep_df = pd.read_csv("../inputs/universe_df_encoded.csv")
```

#### a) Initial PCA


First, we drop non-numerical columns.

```python
drop_cols = [
    "Unnamed: 0",
    "Name",
    "Symbol",
    "Country",
    "Industry Name - GICS Sub-Industry",
    "SEDOL",
    "Instrument",
    "GICS Sector Name",
    "Critical Country 1",
    "Critical Country 2",
    "Critical Country 3",
    "Critical Country 4",
]
cluster_df = prep_df.drop(columns=drop_cols).copy()
```

We then scale our data and perform a 2-dimensional PCA to be able to visualise it later.

```python
features = np.array(cluster_df)

# Scale
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# PCA
pca = PCA(2)
pca_features = pca.fit_transform(scaled_features)
```

We add the PCA columns to the full dataframe, as well as the Refinitiv scores (post PCA.)

```python
# Add columns to full df
prep_df["PCA_1"] = pd.Series(pca_features[:,0])
prep_df["PCA_2"] = pd.Series(pca_features[:,1])

# Letters
prep_df["ESG Score Grade"] = initial_df.loc[:,"ESG Score Grade"]
prep_df['Environmental Pillar Score Grade'] = initial_df.loc[:,"Environmental Pillar Score Grade"]
prep_df['Social Pillar Score Grade'] = initial_df.loc[:,"Social Pillar Score Grade"]
prep_df['Governance Pillar Score Grade'] = initial_df.loc[:,"Governance Pillar Score Grade"]

# Scores
prep_df['ESG Score'] = initial_df.loc[:,"ESG Score"]
prep_df['Environmental Pillar Score'] = initial_df.loc[:,"Environmental Pillar Score"]
prep_df['Social Pillar Score'] = initial_df.loc[:,"Social Pillar Score"]
prep_df['Governance Pillar Score'] = initial_df.loc[:,"Governance Pillar Score"]
```

Next, we check for outliers which could throw off our KMeans algorithm.

```python
sns.boxplot(prep_df["PCA_1"])
plt.title("PCA 1 boxplot")
plt.show()
```

```python
sns.boxplot(prep_df["PCA_2"])
plt.title("PCA 2 boxplot")
plt.show()
```

We identify outliers for PCA_1 components over 50 and PCA_2 components under -10.

```python
mask = prep_df["PCA_1"] > 50
drop_rows = list(prep_df[mask].index)
prep_df[mask]
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

We identify some outliers along the x-axis. Let's performe the elbow method to determine the optimal number of clusters.


#### K-means

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

- https://machinelearningmastery.com/clustering-algorithms-with-python/  
- https://www.kdnuggets.com/2019/10/right-clustering-algorithm.html

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

```python
# define the model
model = AffinityPropagation(damping=0.9)
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
plt.title("Affinity Propagation.")
plt.show()
```

#### Agglomerative clustering

```python
# define the model
model = AgglomerativeClustering(n_clusters=4)
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

```python
# define the model
model = Birch(threshold=0.01, n_clusters=2)
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

```python
# define the model
model = DBSCAN(eps=0.30, min_samples=9)
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

#### Mean shift clustering

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

#### Spectral clustering

```python
# define the model
model = SpectralClustering(n_clusters=2)
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

```python
# define the model
model = GaussianMixture(n_components=2)
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

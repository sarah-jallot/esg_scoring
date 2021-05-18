<!-- #region -->
# Data exploration and initial clustering.
In this notebook, our reference universe will correspond to all authorized stocks within the **Natixis Investment Managers Challenge, 2021 edition**. It comprises over **2,000 companies** for which we retrieved ESG data for year 2020. 
  

 
Our goal is to use **Sustainable Finance Disclosure Regulation (SFDR)** metrics to perform clustering on these companies without using Refinitiv's own notation. Ultimately, we would like to build our own explainable score for the data.
<!-- #endregion -->

First, run all necessary imports.

```python
import pandas as pd
import numpy as np
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
    
def scatterplot(df, x_axis, y_axis, hue, title):
    sns.scatterplot(
        data = df, 
        x = df.loc[:,x_axis] , 
        y = df.loc[:,y_axis] ,
        palette="Set1",
        hue = hue,)
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.show()
    
def corrplot(df, vmax=1, title="Correlation matrix for SFDR metrics"):
    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    f, ax = plt.subplots(figsize=(11, 9))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    
    sns.heatmap(corr, 
                mask=mask, 
                cmap=cmap,
                vmax=vmax,
                center=0,
                square=True, 
                linewidths=.5, 
                cbar_kws={"shrink": .5})
    plt.title(title)
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

```python
corrplot(df, vmax=1, title="Correlation matrix for SFDR metrics")
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
    "Environmental Pillar Score",
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
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
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
    

def kpca_selection(cluster_df, n_components=200, kernel="rbf", threshold=0.5, method="cumsum"):
    """
    Kernel PCA algorithm for feature selection. 
    """
    features = np.array(cluster_df)
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    kpca = KernelPCA(n_components, kernel=kernel)
    kpca_features = kpca.fit_transform(scaled_features)
    
    explained_variance = np.var(kpca_features, axis=0)
    explained_variance_ratio = explained_variance / np.sum(explained_variance)
    
    if method == "cumsum":
        cumsum = np.cumsum(explained_variance_ratio)
        last_feature = np.where(cumsum > threshold)[0][0]-1
        return kpca_features, kpca, kpca_features[:,:last_feature]
        
    if method == "feature-wise":
        mask = explained_variance_ratio_ > threshold
        return kpca_features, kpca, kpca_features[:,np.where(mask == True)[0]]

    

def random_forest_selection(X_train,X_test, y_train, y_test, threshold=0.3):
    """
    Feature selection by using a random forest regressor.
    """
    rf_reg = RandomForestRegressor(n_estimators=100)
    rf_reg.fit(X_train,y_train)
    y_pred_full = rf_reg.predict(X_test)
    r2_full = r2_score(y_test, y_pred_full)
    
    importances = pd.DataFrame(list(cluster_df.columns))
    importances["feature_importances"] = rf_reg.feature_importances_
    importances.columns = ["features", "importances"]
    importances = importances.sort_values(by=["importances"],ascending=False).reset_index().copy()
    

    mask = importances["importances"] > threshold
    n_features = len(importances[mask])
    importances[mask].loc[:,"features":"importances"].plot(kind="barh")
    plt.yticks(ticks=range(n_features),labels=importances[:n_features]["features"])
    plt.title(f"Top {n_features} feature importance for threshold of {threshold}")
    plt.show()
    
    X_important = X_train.loc[:,importances[mask].features]
    rf_reg_important = RandomForestRegressor(n_estimators=100)
    rf_reg_important.fit(X_important,y_train)
    y_pred_imp = rf_reg_important.predict(X_test.loc[:,importances[mask].features])
    r2_imp = r2_score(y_test, y_pred_imp)
    
    return importances, importances[mask], r2_full, r2_imp
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
    "ESG Score",
    "Environmental Pillar Score", 
    "Social Pillar Score", 
    "Governance Pillar Score" 
]
cluster_df = prep_df.drop(columns=drop_cols).copy()
```

```python
cluster_df.to_csv("../inputs/universe_clusters.csv")
```

```python
threshold = 0.8
method = "cumsum"
```

We then scale our data to be able to perform a PCA on it using scikit learn.

```python
full_pca_features, pca, selected_features = pca_selection(cluster_df, threshold=threshold, method=method)
print(f"We selected {selected_features.shape[1]} out of {full_pca_features.shape[1]} features for the {method} method with a threshold of {threshold} .")
```

```python
ticks = list(range(pca.n_components))[::10]
labels = [x for x in ticks]

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
prep_df["PCA_1"] = pd.Series(full_pca_features[:,0])
prep_df["PCA_2"] = pd.Series(full_pca_features[:,1])
scatterplot(prep_df, x_axis="PCA_1", y_axis="PCA_2", title="PCA scatterplot", hue=None)
```

```python
scatterplot(prep_df, x_axis="PCA_1", y_axis="PCA_2", title="PCA scatterplot by sector", hue="GICS Sector Name")
```

```python
X_pca = selected_features
```

### Kernel PCA 
Applying a kernel to make the data linearly separable by PCA. 

```python
threshold = 0.8
n_components = 200
kernel = "rbf"
method = "cumsum"
```

```python
full_kpca_features, kpca, selected_kpca_features = kpca_selection(cluster_df, n_components=n_components, kernel=kernel, threshold=threshold, method=method)
print(f"We selected {selected_kpca_features.shape[1]} out of {full_kpca_features.shape[1]} features for the {method} method with a threshold of {threshold} .")
```

```python
ticks = list(range(kpca.n_components))[::10]
labels = [x for x in ticks]
explained_variance = np.var(full_kpca_features, axis=0)
explained_variance_ratio = explained_variance / np.sum(explained_variance)

figure(figsize=(10,4))
plt.plot(explained_variance,)
plt.xlabel("Number of components")
plt.ylabel("Explained variance")
plt.xticks(ticks=ticks, labels=labels)
plt.title("Kernel PCA feature selection")
plt.show()
```

```python
figure(figsize=(10,4))
plt.plot(explained_variance_ratio)
plt.xlabel("Number of components")
plt.ylabel("Explained variance %")
plt.xticks(ticks=ticks, labels=labels)
plt.title("Kernel PCA feature selection")
plt.show()
```

```python
plt.scatter(x=full_kpca_features[:,0], y=full_kpca_features[:,1])
plt.title("Kernel PCA feature scatterplot")
plt.show()
```

```python
prep_df["KPCA_1"] = pd.Series(full_kpca_features[:,0])
prep_df["KPCA_2"] = pd.Series(full_kpca_features[:,1])
```

```python
X_kpca = selected_kpca_features
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
```

```python
print(f"Full R2 score: {r2_full} versus score after feature selection: {r2_imp}.")
```

```python
mask = importances.importances > threshold
X_rf = cluster_df.loc[:,importances[mask].features]
X_rf.head()
```

```python
corrplot(X_rf, vmax=1, title="Correlation matrix for Random Forest selected features")
```

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

```python

```

### Clustering


As such, cluster analysis is an iterative process where subjective evaluation of the identified clusters is fed back into changes to algorithm configuration until a desired or appropriate result is achieved.

```python
#X = X_pca
X = X_kpca
#X = np.array(X_rf)
```

#### K-means


Assign examples to each cluster while trying to minimize the variance within each cluster. Let's perform the elbow method to determine the optimal number of clusters.

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
    kmeans.fit(X)
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
optimal_nb = 5
```

```python
optimal_kmeans = KMeans(n_clusters=optimal_nb, **kmeans_kwargs)
optimal_kmeans.fit(X)
```

```python
prep_df["kmean_labels"] = optimal_kmeans.labels_
prep_df.head()
```

Now, let's plot our datapoints for each kmeans label.

```python
scatterplot(
    df=prep_df, 
    x_axis="KPCA_1", 
    y_axis="KPCA_2", 
    hue="kmean_labels", 
    title="KPCA coloured by Kmeans"
)
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
mkmeans_model = MiniBatchKMeans(n_clusters=optimal_nb)
# fit the model
mkmeans_model.fit(X)
# assign a cluster to each example
yhat = mkmeans_model.predict(X)
```

```python
prep_df["mkmean_labels"] = mkmeans_model.labels_
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
# define the model
aff_model = AffinityPropagation(damping=0.5)
# fit the model
aff_model.fit(X)
# assign a cluster to each example
yhat = aff_model.predict(X)
```

```python
prep_df["aff_labels"] = aff_model.labels_
```

```python
scatterplot(
    prep_df, 
    "KPCA_1", 
    "KPCA_2", 
    "aff_labels", 
    "KPCA coloured by Mini-kmeans"
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
```

```python
scatterplot(
    prep_df, 
    "KPCA_1", 
    "KPCA_2", 
    "agg_labels", 
    "KPCA coloured by Mini-kmeans"
)
```

#### BIRCH Clustering


Constructing a tree structure from which cluster centroids are extracted.

```python
threshold = 0.01
n_clusters = optimal_nb
```

```python
# define the model
birch_model = Birch(threshold=threshold, n_clusters=n_clusters)
# fit the model
birch_model.fit(X)
# assign a cluster to each example
yhat = birch_model.predict(X)
```

```python
prep_df["birch_labels"] = birch_model.labels_
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
min_samples = 9
```

```python
# define the model
db_model = DBSCAN(eps=eps, min_samples=min_samples)
# fit model and predict clusters
yhat = db_model.fit_predict(X)
```

```python
prep_df["dbscan_labels"] = db_model.labels_
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
mshift_model = MeanShift()
# fit model and predict clusters
yhat = mshift_model.fit_predict(X)
```

```python
prep_df["mshift_labels"] = mshift_model.labels_
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

### OPTICS


Modified version of DBSCAN. 

```python
optics_model = OPTICS(eps=0.8, min_samples=10)
# fit model and predict clusters
yhat = optics_model.fit_predict(X)
```

```python
prep_df["optics_labels"] = optics_model.labels_
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


Here, one uses the top eigenvectors of a matrix derived from the distance between points.

```python
# define the model
spec_model = SpectralClustering(n_clusters=optimal_nb)
# fit model and predict clusters
yhat = spec_model.fit_predict(X)
```

```python
prep_df["spectral_labels"] = spec_model.labels_
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
# define the model
gmm_model = GaussianMixture(n_components=optimal_nb)
# fit the model
gmm_model.fit(X)
```

```python
prep_df["gaussian_labels"] = gmm_model.predict(X)
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

```python
filename = "/universe_df_encoded.csv"
```

```python
df = pd.read_csv(input_path+filename)
```

```python
drop_cols = [
    "ESG Score", 
    "Environmental Pillar Score", 
    "Social Pillar Score", 
    "Governance Pillar Score", 
    "Unnamed: 0", 
    "Name", 
    "Symbol", 
    "Country", # encoded
    "SEDOL",
    "ISINS",
    "GICS Sector Name", # encoded
]
```

```python
y = df.loc[:,"ESG Score": "Governance Pillar Score"].copy()
X = df.drop(columns=drop_cols).copy()
```

```python
X
```

```python
for col in X.columns:
    print(col)
```

```python
#scatterplot(prep_df, x_axis="PCA_1", y_axis="PCA_2", title="PCA scatterplot by sector", hue="ESG Score Grade")
#scatterplot(prep_df, x_axis="PCA_1", y_axis="PCA_2", title="PCA scatterplot by sector", hue="Environmental Pillar Score Grade")
#scatterplot(prep_df, x_axis="PCA_1", y_axis="PCA_2", title="PCA scatterplot by sector", hue="Social Pillar Score Grade")
#scatterplot(prep_df, x_axis="PCA_1", y_axis="PCA_2", title="PCA scatterplot by sector", hue="Governance Pillar Score Grade")
```

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

prep_df["ESG categories"] = prep_df["ESG Score Grade"].map(grades_dict)
prep_df["Environmental categories"] = prep_df["Environmental Pillar Score Grade"].map(grades_dict)
prep_df["Social categories"] = prep_df["Social Pillar Score Grade"].map(grades_dict)
prep_df["Governance categories"] = prep_df["Governance Pillar Score Grade"].map(grades_dict)
```

```python
scatterplot(
    prep_df, 
    x_axis="PCA_1", 
    y_axis="PCA_2", 
    title="PCA scatterplot by ESG Category", 
    hue="ESG Categories"
)
```

```python
scatterplot(
    prep_df, 
    x_axis="PCA_1", 
    y_axis="PCA_2", 
    title="PCA scatterplot by Environmental Category", 
    hue="Environmental Categories"
)
```

```python
scatterplot(
    prep_df, 
    x_axis="PCA_1", 
    y_axis="PCA_2", 
    title="PCA scatterplot by Social Category", 
    hue="Social Categories"
)
```

```python
scatterplot(
    prep_df, 
    x_axis="PCA_1", 
    y_axis="PCA_2", 
    title="PCA scatterplot by Governance Category", 
    hue="Governance Categories"
)
```

```python
#fig,axes =plt.subplots(10,3, figsize=(12, 9)) # 3 columns each containing 10 figures, total 30 features
#excellent= prep_df.loc[prep_df["ESG categories"]=="A"] # define malignant
#bad=prep_df.loc[prep_df["ESG categories"]=="D"] # define benign
#ax=axes.ravel()# flat axes with numpy ravel
#for i in range(30):
#  _,bins=np.histogram(prep_df.iloc[:,i],bins=40)
#  ax[i].hist(excellent.iloc[:,i],bins=bins,color='r',alpha=.5)# red color for malignant class
#  ax[i].hist(bad.iloc[:,i],bins=bins,color='g',alpha=0.3)# alpha is           for transparency in the overlapped region 
#  ax[i].set_title(prep_df.columns[i],fontsize=9)
#  ax[i].axes.get_xaxis().set_visible(False) # the x-axis co-ordinates are not so useful, as we just want to look how well separated the histograms are
#  ax[i].set_yticks(())
#ax[0].legend(['excellent','bad'],loc='best',fontsize=8)
#plt.tight_layout()# let's make good plots
#plt.show()
```

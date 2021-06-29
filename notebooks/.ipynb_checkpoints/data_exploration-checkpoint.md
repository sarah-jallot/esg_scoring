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

import sys
sys.path.append("../utils/")
from utils import *

from sklearn.preprocessing import StandardScaler
#Importing required modules
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from sklearn.ensemble import RandomForestRegressor
from collections import Counter
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

# Data exploration


We exclude ESG notations by Refinitiv from our data exploration, as we will use them only later for verification purposes.  
Note that Refinitiv adopts a **best-in-class** methodology, meaning that our clusters may not correspond to theirs as we run our analysis for all fields together. 


First, download your data.

```python
input_path = "../inputs/"
input_filename = "universe_df_esg.csv"
output_path = "../outputs/"
```

```python
initial_df = pd.read_csv(input_path+input_filename).drop(columns=["Unnamed: 0"])
print(initial_df.shape)
initial_df.head()
```

#### a) SFDR metrics


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
    'TR.AnalyticTotalRenewableEnergy':"Energy Efficiency",
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

To comply with SFDR requirements, we are missing data on:  
- Biodiversity. Red List species / adjacent to sites. 
- Deforestation
- Water stress, untreated discharged waste water
- Due diligence on human rights, human trafficking  
- Number of convictions for anti-corruption


Before we deep-dive into the data, let's study our metric pillar distribution: 

```python
data = [pillar_mapping[key] for key in list(initial_df.columns)]
data = list(filter(lambda a: a != "Other", data))
data = list(filter(lambda a: a != "Target", data))
sns.countplot(data, order = dict(Counter(data).most_common()).keys())
plt.title("SFDR Metrics, Count by pillar")
plt.savefig('images/sfdr_distribution.png')
plt.show()
for key in dict(Counter(data)).keys():
    print(f"{key}: {dict(Counter(data))[key]/sum(list(Counter(data).values()))*100:.1f} percent")

```

Most metrics we tracked are Environmental. They represent 50% of all features. Social comes second with ~30% of all metrics, followed by Governance. 
We remove ESG notations and scores from our dataframe for our data exploration. 

```python
# Simply remove ESG notations for our analysis
df = initial_df.loc[:,"Name":"Bribery, Corruption and Fraud Controversies"].copy()
pp = pprint.PrettyPrinter(indent=4)
pp.pprint(list(df.columns))
print(f"Our dataframe presents {df.shape[1]-6} SFDR-related metrics.")
```

```python
df.head()
```

```python
df.shape
```

#### Geographic repartition. 


Let's observe the geographic repartition of our dataset.

```python
countplot(df,"Country", filename="country_distribution.png")
```

The USA is over-represented in our dataset, and we only work on Western companies.  
Consider that for data processing purposes, when retrieving Thompson-Reuters data, we kept the first occurrence of a company, often labelled as USA. This means that companies under the "USA" flag occurred in our initial dataset in other geographical locations. 


#### Field repartition. 

```python
countplot(df,"GICS Sector Name", filename="sector_distribution.png", figsize=(8,8))
```

Here we see Industrials over-represented in our dataset along with Financials. Let's investigate whether the data's ESG category and it's sector name vary together. 

```python
initial_df["ESG Category"] = simplify_categories(initial_df["ESG Score Grade"])
cramers_stat(initial_df, 'ESG Category', 'GICS Sector Name')
```

#### Market Capitalization

```python
columns = ["GICS Sector Name","Market Capitalization (bil) [USD]"]
boxplot(df, columns,  filename="market_cap_sector.png", categorical=True, figsize=(6,6))
```

```python
upper = 100
columns = ["GICS Sector Name","Market Capitalization (bil) [USD]"]
mask = df[columns[1]] < upper
boxplot(df[mask], columns,  filename="market_cap_100_sector.png", categorical=True, figsize=(6,6))
```

```python
upper = 7500
columns = ["GICS Sector Name","Accidents Total"]
mask = df[columns[1]] < upper
boxplot(df[mask], columns,  filename="accidents_sector_7500.png", categorical=True, figsize=(6,6))
```

#### Boxplots by variable / field

```python
countplot(df,"Whistleblower Protection", filename="whistleblower.png", figsize=(6,6))
```

```python
columns = ["GICS Sector Name","Board Gender Diversity, Percent"]
boxplot(df, columns,  filename="board_gender_div_sector.png", categorical=False, figsize=(6,6))
```

```python
columns = ["GICS Sector Name","CO2 Equivalent Emissions Total"]
boxplot(df, columns, filename="CO2_equivalent_total_sector.png", categorical=True, figsize=(10,6))
```

```python
df["GICS Sector Name"].value_counts().index
```

```python
df[df["GICS Sector Name"] == "Industrials"]["CO2 Equivalent Emissions Total"]
```

```python

```

```python
import scipy.stats as stats

df2 = df.copy()

stats.f_oneway(df2[df2['GICS Sector Name'] == 'Industrials']['CO2 Equivalent Emissions Total'],
               df2[df2['GICS Sector Name'] == 'Financials']['CO2 Equivalent Emissions Total'],
               #df2['CO2 Equivalent Emissions Total'][df2['GICS Sector Name'] == 'Consumer Discretionary'],
               #df2['CO2 Equivalent Emissions Total'][df2['GICS Sector Name'] == 'Information Technology'],
               #df2['CO2 Equivalent Emissions Total'][df2['GICS Sector Name'] == 'Health Care'],
               #df2['CO2 Equivalent Emissions Total'][df2['GICS Sector Name'] == 'Materials'],
               #df2['CO2 Equivalent Emissions Total'][df2['GICS Sector Name'] == 'Real Estate'],
               #df2['CO2 Equivalent Emissions Total'][df2['GICS Sector Name'] == 'Communication Services'],
               #df2['CO2 Equivalent Emissions Total'][df2['GICS Sector Name'] == 'Consumer Staples'],
               #df2['CO2 Equivalent Emissions Total'][df2['GICS Sector Name'] == 'Utilities'],
               #df2['CO2 Equivalent Emissions Total'][df2['GICS Sector Name'] == 'Energy'],
              )
```

Energy, Materials and Utilities present a higher average CO2 emissions total than other industries, with quite a few outliers towards the higher extreme.

```python
columns = ["GICS Sector Name","Board Gender Diversity, Percent"]
boxplot(df, columns, filename="board_gender_diversity_sector.png", categorical=True)
```

```python
columns = ["GICS Sector Name","CO2 Equivalent Emissions Indirect, Scope 2"]
boxplot(df, columns, filename="CO2_emissions_scope2_sector.png", categorical=True)
```

```python
columns = ["GICS Sector Name","CO2 Equivalent Emissions Indirect, Scope 3"]
boxplot(df, columns, filename="CO2_emissions_scope3_sector.png",categorical=True)
```

#### Correlation matrix SFDR - ESG Scoring.

```python
corrplot(df, filename="correlation_matrix.png", vmax=1, title="Correlation matrix for SFDR metrics")
```

#### ESG Scoring study 

```python
columns = ["GICS Sector Name", "ESG Score"]
boxplot(initial_df, columns, filename="ESG_score_sector.png", categorical=True)
```

### Data preprocessing

```python
initial_df = pd.read_csv(input_path+input_filename).drop(columns=["Unnamed: 0"])
df = initial_df.loc[:,"Name":"Bribery, Corruption and Fraud Controversies"].copy()
```

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
df["ESG Score Grade"] = initial_df["ESG Score Grade"]
df['Environmental Pillar Score Grade'] = initial_df.loc[:,"Environmental Pillar Score Grade"]
df['Social Pillar Score Grade'] = initial_df.loc[:,"Social Pillar Score Grade"]
df['Governance Pillar Score Grade'] = initial_df.loc[:,"Governance Pillar Score Grade"]
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

We now check that we have few missing values: 

```python
df_fillna.isna().sum()
```

We can now safely drop rows with missing values.

```python
df_fillna = df_fillna.dropna()
```

```python
print(f"We are left with data for {df_fillna.shape[0]} companies.")
```

```python
#df_fillna.to_csv("../inputs/universe_df_no_nans.csv", index=False)
df_fillna_msci = pd.read_csv("../inputs/universe_df_msci_added.csv")
```

```python
df.shape
```

```python
df_fillna_msci.isna().sum()
```

```python
pp.pprint(list(df_fillna.columns))
```

Then, define the columns to encode using a OneHotEncoder. 

```python
categorical_cols = [
   # "Instrument",to keep trace
    "Country",
    "GICS Sector Name",
    #"Industry Name - GICS Sub-Industry",
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

Let's try removing the sub-industry column to reduce data dimensionality:

```python
df_fillna = df_fillna.drop(columns=["Industry Name - GICS Sub-Industry"])
```

```python
df_encoded = one_hot_encode(df_fillna, categorical_cols)
df_encoded_msci = one_hot_encode(df_fillna_msci, categorical_cols)
```

```python
df_encoded.head()
```

```python
df_encoded.to_csv("../inputs/universe_df_encoded.csv", index=False)
df_encoded_msci.to_csv("../inputs/universe_df_encoded_msci.csv", index=False)
```

### Basic clustering without msci data.

```python
prep_df = pd.read_csv("../inputs/universe_df_encoded.csv")
print(prep_df.shape)
prep_df.head()
```

### Dimensionality reduction techniques


Note that PCA works for data that is linearly separable. Given the complexity of our data, this may well not be the case as we will explore here.  
First, we drop non-numerical columns.

```python
drop_cols = [
    "Name",
    "Symbol",
    "Country",
  #  "Industry Name - GICS Sub-Industry",
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
    "KPCA coloured by Mini-kmeans"
)
```

#### BIRCH Clustering


Constructing a tree structure from which cluster centroids are extracted.

```python
threshold = 0.01
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
min_samples = 10#X_rf.shape[0]/10
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
mshift_model = MeanShift()
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
optics_model = OPTICS(eps=0.8, min_samples=10)
# fit model and predict clusters
yhat = optics_model.fit_predict(X)
```

```python
prep_df["optics_labels"] = optics_model.labels_
X_rf["optics_labels"] = optics_model.labels_
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
optimal_nb = 6
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
scatterplot(
    prep_df, 
    "KPCA_1", 
    "KPCA_2", 
    "gaussian_labels", 
    "KPCA coloured by Gaussian Labels"
)
```

```python
X_rf.to_csv("../inputs/X_rf_labelled.csv", index=False)
```

### Cluster interpretation and visualisation. 

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
cluster_boxplot(X_rf, feature_name = continuous_name, clusters=clusters)
```

```python
cluster_catplot(X_rf, feature_name =categorical_name , clusters=clusters)
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


### ESG Scoring data exploration

```python
pred_df = pd.read_csv("../inputs/universe_df_encoded_msci.csv")
print(pred_df.shape)
pred_df.head()
```

```python
countplot(pred_df,"ESG Score Grade", filename="ESG_score_distribution.png")
```

```python
sns.countplot(x="MSCI_rating", hue="GICS Sector Name", data=pred_df)
plt.figsize((10,10))
plt.show()
```

```python
countplot(pred_df,"MSCI_rating", filename="ESG_score_distribution.png")
```

```python
columns = ["GICS Sector Name", "ESG Score"]
boxplot(pred_df, columns, filename="ESG_score_sector.png", categorical=True)
```

```python
columns = ["GICS Sector Name", "MSCI_rating"]
boxplot(pred_df, columns, filename="MSCI_ESG_score_sector.png", categorical=True)
```

Load the data: 

```python
filename = "/universe_df_encoded.csv"
df = pd.read_csv(input_path+filename)
df["kmean_labels"] = prep_df["kmean_labels"]
df["Environmental Pillar Score Grade"] = prep_df["Environmental Pillar Score Grade"]
df["Social Pillar Score Grade"] = prep_df["Social Pillar Score Grade"]
df["Governance Pillar Score Grade"] = prep_df["Governance Pillar Score Grade"]
```

Format the data into train and test: 

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
    "Industry Name - GICS Sub-Industry",
    "Critical Country 1",
    "Critical Country 2",
    "Critical Country 3",
    "Critical Country 4",
]

X, targets = df.drop(columns=drop_cols).copy(), df.loc[:,"ESG Score": "Governance Pillar Score"].copy()
y = targets.loc[:,"ESG Score"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)
```

```python
X_train.dtypes[X_train.dtypes == str]
```

Select relevant features:

```python
importances, masked_importances, r2_full, r2_imp = random_forest_selection(
    X_train, 
    X_test, 
    y_train, 
    y_test, 
    threshold=0.005
)

features = list(masked_importances.features)
features.append("kmean_labels")
X_train, X_test = X_train.loc[:,features],  X_test.loc[:,features]
```

Define your model:

```python
params = {
    "n_estimators":300, 
    "max_depth":20,
}
model = RandomForestRegressor(**params)
```

Fit your model:

```python
model.fit(X_train,y_train)
```

```python
out = pd.DataFrame(y_test).reset_index()
predictions = model.predict(X_test)
out["predictions"] = predictions
cheat = model.predict(X.loc[:,features])
print(f"R2 score : {r2_score(y, cheat):.2f}")
out
```

```python
X["predictions"] = cheat
X["true"] = df["ESG Score"]
print(f"R2 score on test: {r2_score(y_test, predictions):.2f}")
X
```

```python
#scatterplot(prep_df, x_axis="PCA_1", y_axis="PCA_2", title="PCA scatterplot by sector", hue="ESG Score Grade")
#scatterplot(prep_df, x_axis="PCA_1", y_axis="PCA_2", title="PCA scatterplot by sector", hue="Environmental Pillar Score Grade")
#scatterplot(prep_df, x_axis="PCA_1", y_axis="PCA_2", title="PCA scatterplot by sector", hue="Social Pillar Score Grade")
#scatterplot(prep_df, x_axis="PCA_1", y_axis="PCA_2", title="PCA scatterplot by sector", hue="Governance Pillar Score Grade")
```

Let's check whether we accurately predicted the categories:

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

```python

```

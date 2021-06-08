<!-- #region -->
# ESG Score prediction. 
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

```python
# Utils

def confusion_matrix_labels(X, labels, X_test, y_pred, y_test, percent=False):
    """
    Get the confusion matrix by cluster.
    """
    df_test = X_test.copy()
    df_test["is_correct"] = 1 - (y_pred != y_test)*1
    if percent == True: 
        matrix = pd.crosstab(X.loc[X_test.index, labels], df_test["is_correct"]).T
        return matrix / matrix.sum()
    else:
        return pd.crosstab(X.loc[X_test.index, labels], df_test["is_correct"]).T
    
def confusion_matrix_labels_category(X, labels, X_test, y_pred, y_test, percent=False):
    """
    Get the confusion matrix by cluster by category.
    """
    df_test = X_test.copy()
    df_test["is_correct"] = 1 - 1*(y_pred != y_test)*1
    df_test["ESG Category"] = y_test.copy()
    df_test[labels] = X.loc[X_test.index, labels]
    df_test = pd.DataFrame(df_test.groupby(by=["ESG Category",labels,  "is_correct"]).count().loc[:,labels+"_0"])
    if percent == True:
        df_test.columns = ["percent"]
        return df_test / df_test.sum(axis=0)
    else:
        df_test.columns = ["count"]
        return df_test
```

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
input_path = "../inputs/"
x_path = "X_rf_labelled.csv"
y_path = "universe_df_encoded.csv"
```

```python
X = pd.read_csv(input_path+x_path)
y = pd.read_csv(input_path+y_path).loc[:,"ESG Score Grade"]
df = X.copy()
df["ESG Score Grade"] = y.copy()
df["ESG Category"] = simplify_categories(df["ESG Score Grade"])
```

```python
df.head()
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
labels = "mkmean_labels"
```

```python
categorical_countplot(
    df, 
    labels, 
    "ESG Category", 
    ["A","B","C","D"], 
    filename="ESG_cat_mkmeans.png")
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

Here, you can see that our confidence interval for A and D is better than for B and C: A gets confused with B and D with C. This isn't surprising as they are extreme classes, potentially more recognisable.  
However, it would seem that D is he least well predicted class all the same : in one case out of two, it is not recognised as D and classified in C. 

```python
confusion_mat_df(model, y_test, y_pred, percent=True)
```

Let's investigate our predictions by label group:

```python
confusion_matrix_labels(X, "mkmean_labels", X_test, y_pred, y_test, percent=False)
```

If we look at predictions by group, it seems that classes 1 and two are our least well predicted classes.  
On the other hand, classes five, zero and eight present good accuracy. 

```python
confusion_matrix_labels(X, "mkmean_labels", X_test, y_pred, y_test, percent=True)
```

We now investigate the results by ESG category and by cluster. 
We know that we should pay attention to classes 1 and 2 first.

```python
confusion_matrix_labels_category(X, labels, X_test, y_pred, y_test, percent=False)
```

Here we see that cluster 1 made a bad predictor of class A, but a good predictor of class B. Same for cluster 2, it is a bad predictor of class A.  
0 is a bad predictor for class B, but an excellent one for C, so it balances out in the final accuracy. 
Let's train a specific model for clusters 0, 1 and 2.

```python
def train_random_forest(X, y, params, test_size=0.4, with_labels=False, labels="kmean_labels", ):
    X_trunc = X.loc[:, "Fundamental Human Rights ILO UN":"Bribery, Corruption and Fraud Controversies"]
    if with_labels == False:
        X_train, X_test, y_train, y_test = train_test_split(X_trunc, y, test_size=test_size, random_state=0)
    else:
        X_labels = pd.merge(X_trunc, pd.get_dummies(X.loc[:,labels], prefix=labels), left_index=True, right_index=True)
        X_train, X_test, y_train, y_test = train_test_split(X_labels, y, test_size=test_size, random_state=0)

    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    return model, X_train, X_test, y_train, y_test
```

```python
class Model():

    def __init__:
        custom_clusters = [0,1,2]
        
    def train_test(X,y, test_size=0.4):
        return train_test_split(X, y, test_size=test_size, random_state=0)
        
    def train_model(X_train, y_train, labels, params, test_size=0.4):
        models = []
        model = RandomForestClassifier(**params)
        X_train1 = X_train[X_train[labels].isin(custom_clusters) == False].copy()
        y_train1 = y_train.iloc[X_train1.index,:].copy()
        model.fit(X_train1, y_train1)
        models.append(model)
        for cluster in custom_clusters:
            model = RandomForestClassifier(**params)
            mask = (X_train[labels] == cluster)
            X_train2 = X_train[mask].copy()
            y_train2 = y_train.iloc[X_train[mask].index, :].copy()
            model.fit(X_train2, y_train2)
            models.append(model)
        return models
            
    def predict(X_test, y_test, models):
        y_preds = pd.DataFrame(index=X_test.index, columns=["predictions"])
        X_test1 =  X_test[X_test[labels].isin(custom_clusters) == False].copy()
        y_pred1 = models[0].predict(X_test1)
        y_preds.iloc[X_test1.index, :] = y_pred1.copy()
        
        for cluster in custom_clusters:
            index = custom_clusters.index(cluster)
            mask = (X_test[labels] == cluster)
            X_test2 = X_train[mask].copy()
            y_pred2 = models[index].predict(X_test2)
            y_preds.iloc[X_test2.index, :] = y_pred2
        return y_preds    
```

```python
rf_classifier = Model()
```

```python
rf_classifier.train_model()
```

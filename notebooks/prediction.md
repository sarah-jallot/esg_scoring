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
sys.path.append("../models/")
from models import *

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

from sklearn.neighbors import KNeighborsClassifier

pp = PrettyPrinter(indent=2)
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
features = list(X.loc[:,:"x0_USA"].columns)
#features.append("kmean_labels")
pp.pprint(features)
```

#### a) Logistic Regression benchmark

```python
scaler = StandardScaler()
X_logreg = scaler.fit_transform(np.array(X.loc[:,features]))
```

```python
params = {
    "penalty": "l1", #"elasticnet",
  #  "l1_ratio": 0.2, # if elastic net penalty
  #  "solver": "saga", # if elasticnet penalty
   # "class_weight":"balanced",
}
```

```python
logreg = LogReg(X_logreg, y, params)
```

```python
logreg.train_test_split(test_size=0.2)
```

```python
logreg.train_model()
```

```python
logreg.predict()
```

```python
y_test, y_pred_logreg = logreg.y_pred, logreg.y_test
```

```python
confusion_mat_df(logreg, y_test, y_pred_logreg, percent=False)
```

```python
confusion_mat_df(logreg, y_test, y_pred_logreg, percent=True)
```

#### b) KNN benchmark

```python
params = {
    "n_neighbors":15,
    "weights":"uniform",
    "leaf_size":15,
}
```

```python
knn = KNN(X_logreg, y, params)
```

```python
knn.train_test_split(test_size=0.2)
```

```python
knn.train_model()
```

```python
knn.predict()
```

```python
y_test, y_pred_knn = knn.y_pred, knn.y_test
```

```python
confusion_mat_df(knn, y_test, y_pred_knn, percent=False)
```

```python
confusion_mat_df(knn, y_test, y_pred_knn, percent=True)
```

#### b) Random Forests

```python
params = {
    "n_estimators":200, 
    "max_depth":5,
}
```

```python
model, X_train, X_test, y_train, y_test = train_random_forest(X, y, features, params, test_size=0.2)
```

```python
y_pred_rf = model.predict(X_test)
```

```python
confusion_mat_df(model, y_test, y_pred_rf, percent=False)
```

```python
confusion_mat_df(model, y_test, y_pred_rf, percent=True)
```

```python
pp.pprint(list(X.columns[-10:]))
```

```python
labels = list(X.columns[-10:])
for label in labels:
    print(f"Cramer's coeff {label} : {cramers_stat(df, label, 'ESG Category'):.2f} | Corrected coeff {label} : {cramers_corrected_stat(df, label, 'ESG Category'):.2f}")
```

```python
labels = "mkmean_labels"
test_size = 0.2
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
confusion_matrix_labels(X, "mkmean_labels", X_test, y_pred_rf, y_test, percent=False)
```

```python
model, X_train, X_test, y_train, y_test = train_random_forest(
    X, 
    y,
    features,
    params, 
    test_size, 
    with_labels=True, 
    labels=labels,)
```

```python
y_pred_rf_labels = model.predict(X_test)
```

```python
confusion_mat_df(model, y_test, y_pred_rf_labels, percent=False)
```

Here, you can see that our confidence interval for A and D is better than for B and C: A gets confused with B and D with C. This isn't surprising as they are extreme classes, potentially more recognisable.  
However, it would seem that D is he least well predicted class all the same : in one case out of two, it is not recognised as D and classified in C. 

```python
confusion_mat_df(model, y_test, y_pred_rf_labels, percent=True)
```

Let's investigate our predictions by label group:

```python
confusion_matrix_labels(X, labels, X_test, y_pred_rf_labels, y_test, percent=False)
```

If we look at predictions by group, it seems that classes 1 and two are our least well predicted classes.  
On the other hand, classes five, zero and eight present good accuracy. 

```python
confusion_matrix_labels(X, labels, X_test, y_pred_rf_labels, y_test, percent=True)
```

We now investigate the results by ESG category and by cluster. 
We know that we should pay attention to classes 1 and 2 first.

```python
confusion_matrix_labels_category(X, labels, X_test, y_pred_rf_labels, y_test, percent=False)
```

Here we see that cluster 1 made a bad predictor of class A, but a good predictor of class B. Same for cluster 2, it is a bad predictor of class A.  
0 is a bad predictor for class B, but an excellent one for C, so it balances out in the final accuracy. 
Let's train a specific model for clusters 0, 1 and 2.

```python
categorical_countplot(
    df, 
    "mkmean_labels", 
    "ESG Category", 
    ["A","B","C","D"], 
    filename="ESG_cat_dbscan.png")
```

```python
old_labels = "kmean_labels"
features.pop(features.index(old_labels))
new_labels = "mkmean_labels"
features.append(new_labels)
X = df.loc[:,features]
X.columns
```

```python
y = simplify_categories(y)
```

```python
params = {
    "n_estimators":200, 
    "max_depth":5,
}
```

```python
custom_clusters = [2,3,12]
test_size = 0.2
rf_clf = RuleModel(X, y, custom_clusters ,params, labels=new_labels)
```

```python
X.columns
```

```python
rf_clf.train_test_split(test_size=test_size)
```

```python
rf_clf.train_model()
```

```python
rf_clf.predict()
```

```python
X_test, y_test, y_pred_rule = rf_clf.X_test, rf_clf.y_test, rf_clf.y_preds
```

```python
confusion_mat_df(rf_clf, y_test, y_pred_rule)
```

```python
confusion_mat_df(rf_clf, y_test, y_pred_rule, percent=True)
```

#### Cross-validation

```python
from sklearn.model_selection import GridSearchCV
```

```python
input_path = "../inputs/"
x_path = "X_rf_labelled.csv"
y_path = "universe_df_encoded.csv"
```

```python
X = pd.read_csv(input_path+x_path).loc[:, features]
y = pd.read_csv(input_path+y_path).loc[:,"ESG Score Grade"]
```

```python
y = simplify_categories(y)
```

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=0)
```

#### a) Random Forest Classifier

```python
param_grid = {
    "n_estimators": [100,200,300,400,500],
    "max_depth": [5,10,15,20],
    "min_samples_split":[2,5,10,20],
    "max_features": ["auto", "sqrt", "log2"],
    "class_weight":[None, "balanced"]
}


#param_grid = {
#    "n_neighbors":[5,10,15,20,50],
#    "weights":["uniform","distance"],
#    "leaf_size":[15,30,50,75,100]
#}
rf_clf = RandomForestClassifier()
knn = KNeighborsClassifier()
```

```python
#clf = GridSearchCV(knn, param_grid, n_jobs=-1)
clf = GridSearchCV(rf_clf, param_grid, n_jobs=-1)
```

```python
#clf.fit(X_train,y_train)
```

```python
#clf.cv_results_.keys()
```

```python
#clf.best_estimator_
```

```python
best_params = {
    "n_estimators": 400,
    "max_depth": 5,
    "min_samples_split":20,
    "max_features":  "log2",
    "class_weight":None,
    
}

best_rf = RandomForestClassifier(**best_params)
```

```python
best_rf.fit(X_train, y_train)
```

```python
y_pred = best_rf.predict(X_test)
```

```python
confusion_mat_df(best_rf, y_test, y_pred, percent=True)
```

```python
features.append("mkmean_labels")
```

```python
X = pd.read_csv(input_path+x_path).loc[:, features]
```

```python
custom_clusters = [1,2,3]
test_size = 0.2
best_rule_rf_clf = RuleModel(X, y, custom_clusters ,best_params)
```

```python
best_rule_rf_clf.train_test_split(test_size=test_size)
```

```python
best_rule_rf_clf.train_model()
```

```python
best_rule_rf_clf.predict()
```

```python
X_test, y_test, y_pred = best_rule_rf_clf.X_test, best_rule_rf_clf.y_test, best_rule_rf_clf.y_preds
```

```python
confusion_mat_df(best_rule_rf_clf, y_test, y_pred)
```

```python

```

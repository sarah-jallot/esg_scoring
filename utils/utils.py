import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pprint
from sklearn.preprocessing import OneHotEncoder
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


# Utils

## Data visualisation
def boxplot(df, columns, categorical=True, figsize=(10, 8)):
    """
    Draw categorical boxplot for the series of your choice, excluding missing values.
    """
    data = df.loc[:, columns].dropna()
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


def countplot(df, category, figsize=(10, 6)):
    """
    Countplot for the category of your choice.
    """
    fig, ax = plt.subplots(figsize=figsize)
    plt.setp(ax.get_xticklabels(), rotation=45)
    sns.countplot(
        x=category,
        data=df,
        order=df[category].value_counts().index)
    plt.title(f"Countplot by {category}")
    plt.show()


def scatterplot(df, x_axis, y_axis, hue, title):
    sns.scatterplot(
        data=df,
        x=df.loc[:, x_axis],
        y=df.loc[:, y_axis],
        palette="Set1",
        hue=hue,
        legend="full")
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
    df.loc[:, "Critical Country 1":"Critical Country 4"] = df.loc[:, "Critical Country 1":"Critical Country 4"].fillna(
        "None")

    # Fix GICs industry
    df.iloc[306, 7] = "Electronical Equipment"
    df.iloc[1739, 7] = "Biotechnology"

    # Fix NaNs by industry
    for sector in sectors:
        mask = df.loc[:, "GICS Sector Name"] == sector
        # Median strategy
        for feature in median_strategy_cols:
            nan_value = df[mask].loc[:, feature].median()
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
    X = df.loc[:, categorical_cols]
    enc.fit(X)
    temp = pd.DataFrame(enc.transform(X).toarray())
    temp.columns = enc.get_feature_names()
    out = pd.concat([df.reset_index(), temp.reset_index()], axis=1).drop(columns=["index"])
    return out


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
        last_feature = np.where(cumsum > threshold)[0][0] - 1
        return pca_features, pca, pca_features[:, :last_feature]

    if method == "feature-wise":
        mask = pca.explained_variance_ratio_ > threshold
        return pca_features, pca, pca_features[:, np.where(mask == True)[0]]


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
        last_feature = np.where(cumsum > threshold)[0][0] - 1
        return kpca_features, kpca, kpca_features[:, :last_feature]

    if method == "feature-wise":
        mask = explained_variance_ratio_ > threshold
        return kpca_features, kpca, kpca_features[:, np.where(mask == True)[0]]


def random_forest_selection(X_train, X_test, y_train, y_test, threshold=0.3):
    """
    Feature selection by using a random forest regressor.
    """
    rf_reg = RandomForestRegressor(n_estimators=100)
    rf_reg.fit(X_train, y_train)
    y_pred_full = rf_reg.predict(X_test)
    r2_full = r2_score(y_test, y_pred_full)

    importances = pd.DataFrame(list(X_train.columns))
    importances["feature_importances"] = rf_reg.feature_importances_
    importances.columns = ["features", "importances"]
    importances = importances.sort_values(by=["importances"], ascending=False).reset_index().copy()

    mask = importances["importances"] > threshold
    n_features = len(importances[mask])
    importances[mask].loc[:, "features":"importances"].plot(kind="barh")
    plt.yticks(ticks=range(n_features), labels=importances[:n_features]["features"])
    plt.title(f"Top {n_features} feature importance for threshold of {threshold}")
    plt.show()

    X_important = X_train.loc[:, importances[mask].features]
    rf_reg_important = RandomForestRegressor(n_estimators=100)
    rf_reg_important.fit(X_important, y_train)
    y_pred_imp = rf_reg_important.predict(X_test.loc[:, importances[mask].features])
    r2_imp = r2_score(y_test, y_pred_imp)

    return importances, importances[mask], r2_full, r2_imp


# Understand clustering results.

def cluster_boxplot(X_rf, feature_name='CO2 Equivalent Emissions Total', clusters="kmean_labels"):
    """
    Visualise boxplot for this feature by cluster.
    """
    X_rf.boxplot(feature_name, by=[clusters], figsize=(10, 7))
    plt.title(f"Boxplot of {feature_name} by {clusters}.")
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.show()


def cluster_catplot(X_rf, feature_name="Fundamental Human Rights ILO UN", clusters="kmean_labels"):
    """
    Visualise categorical count for this feature by cluster.
    """
    sns.countplot(x=clusters, hue=feature_name, data=X_rf)
    plt.title(f"Catplot of {feature_name} by {clusters}.")
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.show()
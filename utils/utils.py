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

import pandas as pd
import os, json
from msci_esg.ratefinder import ESGRateFinder


## Data visualisation
def boxplot(df, columns, filename, categorical=True, figsize=(10, 8)):
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
        plt.savefig('images/' + filename)
    else:
        sns.boxplot(
            y=columns[1],
            data=data
        )
        plt.savefig('images/' + filename)


def countplot(df, category, filename, figsize=(10, 6)):
    """
    Countplot for the category of your choice.
    """
    fig, ax = plt.subplots(figsize=figsize)
    plt.setp(ax.get_xticklabels(), rotation=45)
    chart = sns.countplot(
        x=category,
        data=df,
        order=df[category].value_counts().index)
    for p in chart.patches:
        chart.annotate("%.0f" % p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()),
                       ha='center', va='center', fontsize=10, color='black', xytext=(0, 5),
                       textcoords='offset points')
    plt.title(f"Countplot by {category}")
    plt.savefig('images/'+filename)
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


def corrplot(df, filename, vmax=1, title="Correlation matrix for SFDR metrics"):
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
    plt.savefig('images/' + filename)
    plt.show()


## Data preprocessing
def fillna(df, sectors, median_strategy_cols, conservative_cols, drop_cols):
    """
    Fill missing values for our dataframe.
    """
    df.loc[:, "Critical Country 1":"Critical Country 4"] = df.loc[:, "Critical Country 1":"Critical Country 4"].fillna(
        "None")

    # Fix GICs industry
    subsector_to_sector = {
        'Multi-Sector Holdings': "Financials Sector",
        'Heavy Electrical Equipment': "Industrials",
        'Electronic Equipment & Instruments': "Information Technology",
        'Construction Machinery & Heavy Trucks': "Industrials",
        'Technology Hardware, Storage & Peripherals': "Information Technology",
        'Interactive Media & Services': "Communication Services",
        'Building Products': "Industrials",
        'Diversified Metals & Mining': "Materials",
        'Industrial Conglomerates': "Industrial Conglomerates",
        'Semiconductors': "Information Technology",
        'Application Software': "Information Technology",
        'Trading Companies & Distributors': "Industrials",
        'Biotechnology': "Biotechnology",
        'IT Consulting & Other Services': "Information Technology",
        'Aerospace & Defense': "Industrials",
        'Interactive Home Entertainment': "Communication Home Services",
    }
    df.iloc[df[df["GICS Sector Name"].isna()].index,list(df.columns).index("GICS Sector Name")] = df[df["GICS Sector Name"].isna()]["Industry Name - GICS Sub-Industry"].map(subsector_to_sector)

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
    return df


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


# Get MSCI ESG Ratings data using an online package implemented in selenium
# Utils

def load_df(input_path, input_filename):
    """
    Load the universe dataframe to get the symbols.
    """
    initial_df = pd.read_csv(input_path + input_filename)
   # initial_df = initial_df.drop_duplicates(subset=['ISINS']).reset_index().drop(columns=["index"])
   # initial_df = initial_df.drop_duplicates(subset=['Name']).reset_index()
    initial_df = initial_df.drop(columns=["Unnamed: 0"])#, "Unnamed: 0.1", "index", "Unnamed: 0.1.1"])
    return initial_df


def generate_jsons(symbols, js_timeout=2, output_path='../msci_data/'):
    """
    Gets MSCI ratings for each symbol, stored in a separate json file
    """
    counter = 0
    for symbol in symbols:
        response = ratefinder.get_esg_rating(
            symbol=symbol,
            js_timeout=js_timeout
        )
        response["symbol"] = symbol
        with open(output_path + str(counter) + '.json', 'w') as fp:
            json.dump(response, fp)
        counter += 1


def generate_msci_df(path_to_json="../msci_data"):
    """
    Reads all the scraped json files from MSCI ESG Research and returns a dataframe with one row per company.
    """
    json_files = [pos_json for pos_json in os.listdir(path_to_json) if pos_json.endswith('.json')]

    jsons_data = pd.DataFrame(columns=['Symbol', 'MSCI_rating', 'MSCI_category', 'MSCI_history'])

    for index, js in enumerate(json_files):
        with open(os.path.join(path_to_json, js)) as json_file:
            json_text = json.load(json_file)

        symbol = json_text.get("symbol", "N/A")

        if json_text.get("current") == None:
            rating = json_text.get("current", None)
            category = json_text.get("current", None)
        else:
            rating = json_text["current"].get("esg_rating", None)
            category = json_text["current"].get("esg_category", None)
        history = json_text.get("history", None)
        jsons_data.loc[index] = [symbol, rating, category, history]
    return jsons_data


def add_msci(initial_df, path_to_json="../msci_data"):
    jsons_data = generate_msci_df(path_to_json=path_to_json)
    initial_df_msci = pd.merge(initial_df, jsons_data, how="left", left_on="Symbol", right_on=["Symbol"])

    clean_dicts = clean_dictionaries(initial_df_msci)
    df = expand_history(clean_dicts)
    out = pd.merge(
        initial_df_msci,
        df,
        left_index=True,
        right_index=True)

    out.to_csv("../inputs/universe_df_msci_added.csv")
    return out


def clean_dictionaries(initial_df_msci):
    """
    """
    return initial_df_msci['MSCI_history'].apply(
        lambda x: dict(zip([key[-2:] for key in x.keys()], [x[key] for key in x.keys()])) if type(x) == dict else None
    )


def expand_history(clean_dicts):
    df = pd.DataFrame(columns=[
        "MSCI_ESG_2016",
        "MSCI_ESG_2017",
        "MSCI_ESG_2018",
        "MSCI_ESG_2019",
        "MSCI_ESG_2020"])

    for index, my_dict in enumerate(clean_dicts):
        if my_dict == None:
            df.loc[index] = [None, None, None, None, None]
        else:
            msci_2016 = my_dict.get("16", None)
            msci_2017 = my_dict.get("17", None)
            msci_2018 = my_dict.get("18", None)
            msci_2019 = my_dict.get("19", None)
            msci_2020 = my_dict.get("20", None)
            df.loc[index] = [msci_2016, msci_2017, msci_2018, msci_2019, msci_2020]
    return df
    #source venv/bin/activate
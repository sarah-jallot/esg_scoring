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
import scipy.stats as ss

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
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    if categorical == True:
        my_order = data.groupby(by=[columns[0]])[columns[1]].median().iloc[::-1].index
        sns.boxplot(
            x=columns[0],
            y=columns[1],
            data=data,
            order=my_order
        )
        plt.savefig('images/' + filename, bbox_inches='tight')
    else:
        sns.boxplot(
            y=columns[1],
            data=data,
        )
        plt.savefig('images/' + filename, bbox_inches='tight')


def countplot(df, category, filename, figsize=(10, 6)):
    """
    Countplot for the category of your choice.
    """
    fig, ax = plt.subplots(figsize=figsize)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    chart = sns.countplot(
        x=category,
        data=df,
        order=df[category].value_counts().index)
    for p in chart.patches:
        chart.annotate("%.0f" % p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()),
                       ha='center', va='center', fontsize=10, color='black', xytext=(0, 5),
                       textcoords='offset points')
    plt.title(f"Countplot by {category}")
    plt.savefig('images/'+filename, bbox_inches='tight')
    plt.show()


def categorical_countplot(df, x_axis="kmean_labels", hue="ESG Category", hue_order=["A", "B", "C", "D"], filename="ESG_cat_kmeans.png"):
    plt.figure(figsize=(17, 8))
    chart = sns.countplot(
        x=x_axis,
        hue=hue,
        data=df,
        palette="husl",
        hue_order= hue_order
    )
    for p in chart.patches:
        chart.annotate("%.0f" % p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()),
                       ha='center', va='center', fontsize=10, color='black', xytext=(0, 5),
                       textcoords='offset points')
    plt.title(f"{hue} repartition by {x_axis} cluster.")
    plt.savefig('images/' + filename, bbox_inches='tight')
    plt.show()

def scatterplot(df, x_axis, y_axis, hue, title, filename):
    sns.scatterplot(
        data=df,
        x=df.loc[:, x_axis],
        y=df.loc[:, y_axis],
        palette="Set1",
        hue=hue,
        legend="full")
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.savefig("images/"+filename, bbox_inches='tight')
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
    plt.savefig('images/' + filename, bbox_inches='tight')
    plt.show()

def catplot(df, columns=["ESG Score Grade", "GICS Sector Name", "Fundamental Human Rights ILO UN"], figsize=(10,0.5), order=[""]):
    sns.catplot(
        x=columns[0],
        hue=columns[1],
        col=columns[2],
        data=df,
        kind="count",
        height=figsize[0],
        aspect=figsize[1],
        order=order)
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
    print(importances[mask].loc[:, "features":"importances"])
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


def generate_jsons(symbols, sedols, js_timeout=2, output_path='../msci_data/'):
    """
    Gets MSCI ratings for each symbol, stored in a separate json file
    """
    ratefinder = ESGRateFinder()
    counter = 0
    for symbol in symbols:
        response = ratefinder.get_esg_rating(
            symbol=symbol,
            js_timeout=js_timeout
        )
        response["symbol"] = symbol
        response["sedol"] = sedols[symbols.index(symbol)]
        with open(output_path + str(counter) + '.json', 'w') as fp:
            json.dump(response, fp)
        counter += 1


def generate_msci_df(path_to_json="../msci_data"):
    """
    Reads all the scraped json files from MSCI ESG Research and returns a dataframe with one row per company.
    """
    json_files = [pos_json for pos_json in os.listdir(path_to_json) if pos_json.endswith('.json')]

    jsons_data = pd.DataFrame(columns=['Symbol', 'SEDOL', 'rating_paragraph', 'MSCI_rating', 'MSCI_category', 'MSCI_history'])

    for index, js in enumerate(json_files):
        with open(os.path.join(path_to_json, js)) as json_file:
            json_text = json.load(json_file)

        symbol = json_text.get("symbol", "N/A")
        sedol = json_text.get("sedol", "N/A")
        rating_paragraph = json_text.get("rating-paragraph", None)

        if json_text.get("current") == None:
            rating = json_text.get("current", None)
            category = json_text.get("current", None)
        else:
            rating = json_text["current"].get("esg_rating", None)
            category = json_text["current"].get("esg_category", None)
        history = json_text.get("history", None)
        jsons_data.loc[index] = [symbol, sedol, rating_paragraph, rating, category, history]
    return jsons_data


def add_msci(initial_df, path_to_json="../msci_data"):
    jsons_data = generate_msci_df(path_to_json=path_to_json)
    print(initial_df.shape, jsons_data.shape)
    initial_df_msci = pd.merge(initial_df, jsons_data, how="left", left_on="SEDOL", right_on=["SEDOL"])
    print(initial_df_msci.shape)

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

# ESG Predictions

# Utils

def plot_pca(pca, percent=False, cumsum=False):
    ticks = list(range(pca.n_components))[::10]
    labels = [x for x in ticks]
    if percent == True:
        if cumsum == True:
            figure(figsize=(10, 4))
            plt.plot(np.cumsum(pca.explained_variance_ratio_))
            plt.xlabel("Number of components")
            plt.ylabel("Cumsum of explained variance %")
            plt.xticks(ticks=ticks, labels=labels)
            plt.title("PCA feature selection, cumsum explained variance in %")
            plt.savefig("images/PCA_feature_selection_cumsum_percent.png")
            plt.show()
        else:
            figure(figsize=(10, 4))
            plt.plot(pca.explained_variance_ratio_)
            plt.xlabel("Number of components")
            plt.ylabel("Explained variance %")
            plt.xticks(ticks=ticks, labels=labels)
            plt.title("PCA feature selection, explained variance in %")
            plt.savefig("images/PCA_feature_selection_percent.png")
            plt.show()
    else:
        if cumsum == True:
            figure(figsize=(10, 4))
            plt.plot(np.cumsum(pca.explained_variance_))
            plt.xlabel("Number of components")
            plt.ylabel("Cumsum of explained variance")
            plt.xticks(ticks=ticks, labels=labels)
            plt.title("PCA feature selection, cumsum explained variance in %")
            plt.savefig("images/PCA_feature_selection_cumsum.png")
            plt.show()
        else:
            figure(figsize=(10, 4))
            plt.plot(pca.explained_variance_, )
            plt.xlabel("Number of components")
            plt.ylabel("Explained variance")
            plt.xticks(ticks=ticks, labels=labels)
            plt.title("PCA feature selection")
            plt.savefig("images/PCA_feature_selection.png")
            plt.show()


def plot_kpca(kpca, kpca_features, percent=False, cumsum=False):
    ticks = list(range(kpca.n_components))[::10]
    labels = [x for x in ticks]
    explained_variance = np.var(kpca_features, axis=0)
    explained_variance_ratio = explained_variance / np.sum(explained_variance)
    if percent == True:
        if cumsum == True:
            figure(figsize=(10, 4))
            plt.plot(np.cumsum(explained_variance_ratio))
            plt.xlabel("Number of components")
            plt.ylabel("Cumsum of explained variance %")
            plt.xticks(ticks=ticks, labels=labels)
            plt.title("KPCA feature selection, cumsum explained variance in %")
            plt.savefig("images/KPCA_feature_selection_cumsum_percent.png")
            plt.show()
        else:
            figure(figsize=(10, 4))
            plt.plot(explained_variance_ratio)
            plt.xlabel("Number of components")
            plt.ylabel("Explained variance %")
            plt.xticks(ticks=ticks, labels=labels)
            plt.title("KPCA feature selection, explained variance in %")
            plt.savefig("images/KPCA_feature_selection_percent.png")
            plt.show()
    else:
        if cumsum == True:
            figure(figsize=(10, 4))
            plt.plot(np.cumsum(explained_variance))
            plt.xlabel("Number of components")
            plt.ylabel("Cumsum of explained variance")
            plt.xticks(ticks=ticks, labels=labels)
            plt.title("KPCA feature selection, cumsum explained variance in %")
            plt.savefig("images/KPCA_feature_selection_cumsum.png")
            plt.show()
        else:
            figure(figsize=(10, 4))
            plt.plot(explained_variance, )
            plt.xlabel("Number of components")
            plt.ylabel("Explained variance")
            plt.xticks(ticks=ticks, labels=labels)
            plt.title("KPCA feature selection")
            plt.savefig("images/KPCA_feature_selection.png")
            plt.show()


def kmeans(X, kmeans_kwargs, categories, upper=10, plot=True):
    """
    Run the kmeans algorithm for various numbers of clusters.
    Plot the elbow graph to find the optimal k.
    X: normalised features to perform clustering on.
    kmeans_kwargs: dictionary containing your kmeans arguments.
    upper: the maximal number of clusters to test.
    plot: boolean indicating whether to plot the elbow graph.

    :returns: the sse as a list.
    """
    # A list holds the SSE values for each k
    sse = []
    cramers = []
    lower = 1
    for k in range(lower, upper):
        kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
        kmeans.fit(X)
        sse.append(kmeans.inertia_)
    if plot == True:
        plt.style.use("fivethirtyeight")
        plt.figure(figsize=(15, 5))
        plt.plot(range(lower + 1, upper + 1), sse)
        plt.xticks(range(lower + 1, upper + 1), rotation=45)
        plt.xlabel("Number of Clusters")
        plt.ylabel("SSE")
        plt.title("Elbow graph for KMeans")
        plt.savefig('images/elbow_graph_kmeans.png')
        plt.show()
    return sse


def m_kmeans(X, upper=10, plot=True):
    """
    Run the kmeans algorithm for various numbers of clusters.
    Plot the elbow graph to find the optimal k.
    X: normalised features to perform clustering on.
    kmeans_kwargs: dictionary containing your kmeans arguments.
    upper: the maximal number of clusters to test.
    plot: boolean indicating whether to plot the elbow graph.

    :returns: the sse as a list.
    """
    # A list holds the SSE values for each k
    sse = []
    lower = 1
    for k in range(lower, upper):
        model = MiniBatchKMeans(n_clusters=k)
        model.fit(X)
        sse.append(model.inertia_)
    if plot == True:
        plt.style.use("fivethirtyeight")
        plt.figure(figsize=(15, 5))
        plt.plot(range(lower + 1, upper + 1), sse)
        plt.xticks(range(lower + 1, upper + 1))
        plt.xlabel("Number of Clusters")
        plt.ylabel("SSE")
        plt.title("Elbow graph for Mini-batch KMeans")
        plt.show()
        plt.savefig('images/elbow_graph_mkmeans.png')
        plt.show()
    return sse

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


def simplify_categories(series):
    return series.str.replace("+", "").str.replace("-", "")

def simplify_msci_categories(series):
    return series.apply(lambda x: x[0])

def train_random_forest(X, y, features, params, test_size=0.4, with_labels=False, labels="kmean_labels", ):
    X_trunc = X.loc[:, features].copy()
    if with_labels == False:
        X_train, X_test, y_train, y_test = train_test_split(X_trunc, y, test_size=test_size, random_state=0)
    else:
        X_labels = pd.merge(X_trunc, pd.get_dummies(X.loc[:,labels], prefix=labels), left_index=True, right_index=True)
        X_train, X_test, y_train, y_test = train_test_split(X_labels, y, test_size=test_size, random_state=0)

    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    return model, X_train, X_test, y_train, y_test


def confusion_mat_df(model, y_test, y_pred, percent=False):
    """
    Format the confusion matrix properly.
    """
    print(f"Accuracy: {100*accuracy_score(y_test, y_pred):.2f}%")
    if percent == False:
        confusion_mat = confusion_matrix(y_test, y_pred)
    else:
        confusion_mat = confusion_matrix(y_test, y_pred) / confusion_matrix(y_test, y_pred).sum(axis=0) * 100
    confusion_mat = pd.DataFrame(confusion_mat)
    confusion_mat.columns = model.classes_
    confusion_mat.index = model.classes_
    return confusion_mat


def confusion_matrix_labels(X, labels, X_test, y_pred, y_test, percent=False):
    """
    Get the confusion matrix by cluster.
    """
    df_test = X_test.copy()
    df_test["is_correct"] = 1 - (y_pred != y_test) * 1
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

# Stats
def cramers_stat(df, cat1, cat2):
    # Chi-squared test statistic, sample size, and minimum of rows and columns
    confusion_matrix = np.array(pd.crosstab(df[cat1], df[cat2]))
    X2 = ss.chi2_contingency(confusion_matrix, correction=False)[0]
    n = np.sum(confusion_matrix)
    phi2 = X2 / n
    minDim = min(confusion_matrix.shape) - 1
    # calculate Cramer's V
    V = np.sqrt((phi2) / minDim)
    return V


def cramers_corrected_stat(df, cat1, cat2):
    """ calculate Cramers V statistic for categorial-categorial association.
        uses correction from Bergsma and Wicher,
        Journal of the Korean Statistical Society 42 (2013): 323-328
    """
    confusion_matrix = np.array(pd.crosstab(df[cat1], df[cat2]))
    X2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum()
    phi2 = X2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - ((r - 1) ** 2) / (n - 1)
    kcorr = k - ((k - 1) ** 2) / (n - 1)
    return np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))
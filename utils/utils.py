import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder

def boxplot(df, columns, categorical=True):
    """
    Draw categorical boxplot for the series of your choice, excluding missing values.
    """
    data = df.loc[:, columns].dropna()
    print(f"Boxplot for {len(data)} datapoints out of {len(df)} overall.")

    fig, ax = plt.subplots(figsize=(10, 8))
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


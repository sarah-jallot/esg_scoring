```python
import pandas as pd
import os, json
from msci_esg.ratefinder import ESGRateFinder
```

```python
import sys 
sys.path.append("../utils/")
from utils import *
```

```python
input_path = "../inputs/"
input_filename = "universe_df_no_nans.csv"#universe_df_full_scores.csv"
output_path = "../outputs/"
path_to_json = "../msci_data/"
```

```python
initial_df = pd.read_csv(input_path+input_filename)
sedols = list(initial_df.SEDOL)
symbols = list(initial_df.Symbol)
```

```python
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
```

```python
generate_jsons(symbols, sedols, js_timeout=1, output_path=path_to_json)
```

First, load our dataframe without MSCI data.

```python
initial_df = load_df(input_path, input_filename)
symbols, sedols = list(initial_df.Symbol), list(initial_df.SEDOL)
generate_jsons(symbols, js_timeout=2, output_path='../msci_data/')
```

Then, from the scraped data which we stored locally, we add the data to our universe dataframe. 

```python
jsons_data = generate_msci_df(path_to_json="../msci_data")
jsons_data.head()
```

```python
initial_df_msci = add_msci(initial_df)
initial_df_msci.head()
```

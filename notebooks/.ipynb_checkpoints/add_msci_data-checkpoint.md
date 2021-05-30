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
# Utils

def load_df(input_path, input_filename):
    """
    Load the universe dataframe to get the symbols. 
    """
    initial_df = pd.read_csv(input_path+input_filename)
    initial_df = initial_df.drop_duplicates(subset=['ISINS']).reset_index().drop(columns=["index"])
    initial_df = initial_df.drop_duplicates(subset=['Name']).reset_index().drop(columns=["Unnamed: 0","Unnamed: 0.1", "index", "Unnamed: 0.1.1"])
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
        with open(output_path+str(counter)+'.json', 'w') as fp:
            json.dump(response, fp)
        counter +=1
        
def generate_msci_df(path_to_json="../msci_data"):
    """
    Reads all the scraped json files from MSCI ESG Research and returns a dataframe with one row per company.
    """
    json_files = [pos_json for pos_json in os.listdir(path_to_json) if pos_json.endswith('.json')]
    
    jsons_data = pd.DataFrame(columns=['Symbol', 'MSCI_rating', 'MSCI_category', 'MSCI_history'])
    
    for index, js in enumerate(json_files):
        with open(os.path.join(path_to_json, js)) as json_file:
            json_text = json.load(json_file)
        
        symbol = json_text.get("symbol","N/A")
        
        if json_text.get("current") == None:
            rating = json_text.get("current",None)
            category = json_text.get("current",None)
        else: 
            rating = json_text["current"].get("esg_rating", None)
            category = json_text["current"].get("esg_category", None)
        history = json_text.get("history",None)
        jsons_data.loc[index] = [symbol, rating, category, history]
    return jsons_data

def add_msci(initial_df, path_to_json = "../msci_data"):
    jsons_data = generate_msci_df(path_to_json="../msci_data")
    out = pd.merge(initial_df, jsons_data, how="left", left_on="Symbol", right_on=["Symbol"])
    
    clean_dicts = clean_dictionaries(initial_df_msci)
    df = expand_history(clean_dicts)
    out_bis = pd.merge(
        out, 
        df, 
        left_index=True, 
        right_index=True)
    
    out.to_csv("../inputs/universe_df_msci_added.csv")
    return out

def clean_dictionaries(initial_df_msci):
    """
    """
    return initial_df_msci['history'].apply(
        lambda x: dict(zip([key[-2:] for key in x.keys()], [x[key] for key in x.keys()])) if type(x)==dict else None
    )

def expand_history(clean_dicts): 
    df = pd.DataFrame(columns=[
    "MSCI_ESG_2016", 
    "MSCI_ESG_2017",
    "MSCI_ESG_2018",
    "MSCI_ESG_2019",
    "MSCI_ESG_2020" ])
    
    for index, my_dict in enumerate(clean_dicts):
        if my_dict == None:
            df.loc[index] = [None, None, None, None, None]
        else:
            msci_2016 = my_dict.get("16", None)
            msci_2017 = my_dict.get("17",None)
            msci_2018 = my_dict.get("18",None)
            msci_2019 = my_dict.get("19",None)
            msci_2020 = my_dict.get("20",None)
        df.loc[index] = [msci_2016, msci_2017, msci_2018, msci_2019, msci_2020]
    return df    
```

```python
input_path = "../inputs/"
input_filename = "universe_df_full_scores.csv"
output_path = "../outputs/"
path_to_json = "../msci_data"
```

First, load our dataframe without MSCI data.

```python
initial_df = load_df(input_path, input_filename)
symbols = initial_df.Symbol
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

```python
initial_df_msci.isna().sum()
```

```python
mask = initial_df_msci["MSCI_rating"].isna()
initial_df_msci[mask]
```

```python

```

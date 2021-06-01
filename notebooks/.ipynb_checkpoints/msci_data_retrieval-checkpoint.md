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
path_to_json = "../msci_data"
```

First, load our dataframe without MSCI data.

```python
pd.read_csv(input_path+input_filename)
```

```python
initial_df = load_df(input_path, input_filename)
symbols = initial_df.Symbol
```

```python
#generate_jsons(symbols, js_timeout=2, output_path='../msci_data/')
```

Then, from the scraped data which we stored locally, we add the data to our universe dataframe. 

```python
jsons_data = generate_msci_df(path_to_json="../msci_data")
jsons_data.head()
```

```python
jsons_data.isna().sum()
```

```python
clean_dicts = clean_dictionaries(jsons_data)
```

```python
clean_dicts.isna().sum()
```

```python
initial_df_msci = add_msci(initial_df)
initial_df_msci.head()
```

```python
initial_df_msci.isna().sum()
```

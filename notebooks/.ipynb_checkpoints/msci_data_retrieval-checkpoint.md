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

First, load our dataframe without MSCI data.

```python
#initial_df = load_df(input_path, input_filename)
#symbols, sedols = list(initial_df.Symbol), list(initial_df.SEDOL)
#generate_jsons(symbols, sedols, js_timeout=1, output_path=path_to_json)
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

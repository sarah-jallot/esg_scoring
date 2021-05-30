```python
import pandas as pd
from msci_esg.ratefinder import ESGRateFinder
```

```python
input_path = "../inputs/"
input_filename = "universe_df_full_scores.csv"
output_path = "../outputs/"
```

```python
initial_df = pd.read_csv(input_path+input_filename)
initial_df = initial_df.drop_duplicates(subset=['ISINS']).reset_index().drop(columns=["index"])
initial_df = initial_df.drop_duplicates(subset=['Name']).reset_index().drop(columns=["Unnamed: 0","Unnamed: 0.1", "index", "Unnamed: 0.1.1"])
initial_df.head()
```

```python
symbols = initial_df.Symbol
```

```python
symbols[0]
```

```python
symbol_list = symbols[:2]
symbol_list
```

```python
# Create an ESGRateFinder object, optionally passing in debug=True for more print statements
ratefinder = ESGRateFinder(
   # debug=True
)
```

```python
for symbol in symbol_list:
    response = ratefinder.get_esg_rating(
        symbol=symbols[0],
        js_timeout=2
    )
    response["symbol"] = symbol
    with open('../msci_data/'+response["symbol"]+'.json', 'w') as fp:
        json.dump(response, fp)
```

```python
# Call the ratefinder object's get_esg_rating method, passing in the Apple stock symbol and 
# a JS timeout of 5 seconds (this is how long the Selenium web driver should wait for JS to execute 
# before scraping content)
response = ratefinder.get_esg_rating(
    symbol=symbols[0],
    js_timeout=5
)
# The response is a dictionary; print it
#
```

```python
response["symbol"] = symbols[0]
```

```python
response.
```

```python
path_to_json = "../msci_data"
json_files = [pos_json for pos_json in os.listdir(path_to_json) if pos_json.endswith('.json')]
```

```python
pd.DataFrame.from_dict("../"+json_files[0])#json_files
```

```python
import json
with open('../'+response["symbol"]+'.json', 'w') as fp:
    json.dump(response, fp)
```

```python
df = pd.read_json("../msci_data/0.json")

for nb in range(2):
    df = df.append(pd.read_json("../msci_data/"+str(nb+1)+".json"))
```

```python
df
```

```python
pd.read_csv("../inputs/universe_df_full_scores.csv")
```

```python

```

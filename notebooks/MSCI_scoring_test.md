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
path_to_json = "../msci_data"
json_files = [pos_json for pos_json in os.listdir(path_to_json) if pos_json.endswith('.json')]
```

```python
import os, json
import pandas as pd

# this finds our json files
path_to_json = 'json/'
json_files = [pos_json for pos_json in os.listdir(path_to_json) if pos_json.endswith('.json')]

# here I define my pandas Dataframe with the columns I want to get from the json
jsons_data = pd.DataFrame(columns=['country', 'city', 'long/lat'])

# we need both the json and an index number so use enumerate()
for index, js in enumerate(json_files):
    with open(os.path.join(path_to_json, js)) as json_file:
        json_text = json.load(json_file)

        # here you need to know the layout of your json and each json has to have
        # the same structure (obviously not the structure I have here)
        country = json_text['features'][0]['properties']['country']
        city = json_text['features'][0]['properties']['name']
        lonlat = json_text['features'][0]['geometry']['coordinates']
        # here I push a list of data into a pandas DataFrame at row given by 'index'
        jsons_data.loc[index] = [country, city, lonlat]

# now that we have the pertinent json data in our DataFrame let's look at it
print(jsons_data)
```

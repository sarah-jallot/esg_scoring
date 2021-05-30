import pandas as pd
import json
from msci_esg.ratefinder import ESGRateFinder

input_path = "../../inputs/"
input_filename = "universe_df_full_scores.csv"
output_path = "../outputs/"

initial_df = pd.read_csv(input_path+input_filename)
initial_df = initial_df.drop_duplicates(subset=['ISINS']).reset_index().drop(columns=["index"])
initial_df = initial_df.drop_duplicates(subset=['Name']).reset_index().drop(columns=["Unnamed: 0","Unnamed: 0.1", "index", "Unnamed: 0.1.1"])

symbols = initial_df.Symbol

# Create an ESGRateFinder object, optionally passing in debug=True for more print statements
ratefinder = ESGRateFinder(
    debug=True
)

# Call the ratefinder object's get_esg_rating method, passing in the Apple stock symbol and
# a JS timeout of 5 seconds (this is how long the Selenium web driver should wait for JS to execute
# before scraping content)
counter = 0
for symbol in symbols:
    response = ratefinder.get_esg_rating(
        symbol=symbol,
        js_timeout=2
    )
    response["symbol"] = symbol
   # response["symbol"] = symbol
    with open('../../msci_data/'+str(counter)+'.json', 'w') as fp:
        json.dump(response, fp)
    counter +=1
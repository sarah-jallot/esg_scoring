import sys
#sys.path.insert(0,"../../")

import json
import pandas as pd
from utils.utils import *

import os
from msci_esg.ratefinder import ESGRateFinder


input_path = "inputs/"
input_filename = "universe_df_no_nans.csv"
df = load_df(input_path, input_filename)
symbols, sedols = list(df.Symbol), list(df.SEDOL)
generate_jsons(symbols[:5], sedols[:5], js_timeout=2, output_path='../msci_data/')
import read_data
import pandas as pd

body_df = read_data.get_body_df(n_articles=10)

for i in body_df["body"]:
    print(len(i))

# TODO some visuals

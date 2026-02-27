import os
import pandas as pd
from icecream import ic
from functools import partial

sample_dir = "./samples"
os.makedirs(sample_dir, exist_ok=True)

time_range = pd.date_range(start="2022-10-01", end="2023-09-30", freq="6H")
df = pd.DataFrame({"date": time_range})
to_datetime_fmt = partial(pd.to_datetime, format="%y-%m-%d %H:%M:%S", unit="s")
df["time_start"] = df["date"].apply(to_datetime_fmt)
df.index = df["date"]
df = df.sort_index()
df.to_csv(f"./{sample_dir}/test_index.csv")

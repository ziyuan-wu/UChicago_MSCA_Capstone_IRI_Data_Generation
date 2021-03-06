import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt


df = pd.read_csv('~/IRI/store_drug_combined_total.csv', index_col=0)


df['F'] = df['F'].astype("str")
df['D'] = df['D'].astype("str")
df['PR'] = df['PR'].astype("str")

from sdv.tabular import CTGAN

import timeit

start = timeit.default_timer()


model = CTGAN()
model.fit(df)


stop = timeit.default_timer()

print('Time: ', stop - start)  

new_data = model.sample(num_rows=200)

model.save('~/IRI/simple_model.pkl')

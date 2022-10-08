import csv
import pandas as pd
import matplotlib.pyplot as plt
from pandas_datareader import data as pdr

label = "ERJ"
#label = "GS"
output_steps = 52

result = pdr.get_data_yahoo(label, start="2002-03-22")[['Adj Close']].rename(columns={'Adj Close': label})
result = result.loc[result.index.day_name() == 'Friday', :]
result = result[-output_steps * 5:]

with open(f"results/{label}_NARX_prediction.csv", 'r') as csvfile:
    reader = csv.reader(csvfile)
    data = list(map(float, list(reader)[0]))

result[label + ' (NARX)'] = pd.DataFrame({f'{label} (NARX)': data},
                                         index=result[-output_steps:].index)

result.plot()
plt.show()

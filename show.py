import csv
import pandas as pd
import matplotlib.pyplot as plt
from pandas_datareader import data as pdr
from sklearn.metrics import mean_absolute_error
from util import save_figure, get_test_curve

label = "ERJ"
# label = "GS"
output_steps = 12

if label == "ERJ":
    result = pdr.get_data_yahoo(label, start="2002-03-22")[['Adj Close']].rename(columns={'Adj Close': label})
elif label == "GS":
    result = pdr.get_data_yahoo(label, start="1999-05-07")[['Adj Close']].rename(columns={'Adj Close': label})

result = result.resample('1D').interpolate()
if output_steps == 12:
    result = result.loc[result.index.is_month_end, :]
elif output_steps == 52:
    result = result.loc[result.index.day_name() == 'Friday', :]
result = result[-output_steps * 5:]

with open(f"results/{label}_NARX_prediction.csv", 'r') as csvfile:
    reader = csv.reader(csvfile)
    data = list(map(float, list(reader)[0]))

result[label + ' (NARX)'] = pd.DataFrame({f'{label} (NARX)': data},
                                         index=result[-output_steps:].index)

result.plot()
plt.title(f"mean absolute error: {mean_absolute_error(result[label][-output_steps:], data)}")

if output_steps == 12:
    save_figure(f"{label}_NARX_prediction_monthly")
elif output_steps == 52:
    save_figure(f"{label}_NARX_prediction_weekly")
else:
    print("Error: output_steps must be 12 or 52")


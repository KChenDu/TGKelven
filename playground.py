import yfinance as yf
import pandas as pd
from pandas_datareader import data as pdr
from util import *
from sklearn.ensemble import RandomForestRegressor
from fireTS.models import NARX
from sklearn.metrics import mean_absolute_error

if __name__ == "__main__":
    yf.pdr_override()

    #mktdata = pdr.get_data_yahoo("ERJ", start="2002-03-21")[['Adj Close']].rename(columns={'Adj Close': 'ERJ'})
    #mktdata['PBR'] = pdr.get_data_yahoo("PBR", start="2002-03-21")['Adj Close']
    #mktdata['VALE'] = pdr.get_data_yahoo("VALE")['Adj Close']
    mktdata = pdr.get_data_yahoo("GS", start="2002-03-21")[['Adj Close']].rename(columns={'Adj Close': 'GS'})
    mktdata['JPM'] = pdr.get_data_yahoo('JPM', start="2002-03-21")['Adj Close']
    mktdata['AXP'] = pdr.get_data_yahoo("AXP")['Adj Close']
    mktdata = mktdata.loc[mktdata.index.day_name() == 'Friday', :]
    mktdata.plot()
    plt.show()

    label = 'GS'
    output_steps = 52
    auto_order = 52
    exog_order = [52, 52]

    train_df = mktdata[:-output_steps]
    result = mktdata[-output_steps * 5:]

    normalizer = Normalizer(train_df[label])  # Caution: Use series input, not dataframe

    narx = NARX(RandomForestRegressor(), auto_order, exog_order)
    train_x = normalizer.normalize(train_df.loc[:, train_df.columns != label])
    train_y = normalizer.normalize(train_df[label])
    narx.fit(train_x, train_y)
    x = normalizer.normalize(mktdata.loc[:, mktdata.columns != label])
    y = normalizer.normalize(mktdata[label])
    output = pd.DataFrame({label + ' (NARX)': narx.predict(x, y, output_steps)[-output_steps:]}, index=mktdata[-output_steps:].index)
    output = normalizer.denormalize(output)
    result = result.join(output)
    result.plot()
    plt.show()

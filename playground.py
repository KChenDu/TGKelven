from pandas_datareader import data as pdr
from matplotlib import pyplot as plt

mktdata = pdr.get_data_yahoo("AAPL", start="2002-03-22")[['Adj Close']].rename(columns={'Adj Close': 'AAPL'})
mktdata['C'] = pdr.get_data_yahoo("C", start="2002-03-22")['Adj Close']
mktdata['BBY'] = pdr.get_data_yahoo("BBY", start="2002-03-22")['Adj Close']
mktdata['AAL'] = pdr.get_data_yahoo("BBY", start="2002-03-22")['Adj Close']

mktdata.plot()
plt.show()

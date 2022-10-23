from pandas_datareader import data as pdr
from util import get_test_curve, add_trigonometric_input, save_figure

'''test curve'''
mktdata = get_test_curve(480, 150, 100)
save_figure('y')
mktdata = add_trigonometric_input(mktdata)
mktdata.to_csv('datas/y.csv')

'''PBR curve'''
mktdata = pdr.get_data_yahoo("ERJ", start="2002-03-22")[['Adj Close']].rename(columns={'Adj Close': 'ERJ'})
mktdata['PBR'] = pdr.get_data_yahoo("PBR", start="2002-03-22")['Adj Close']
mktdata['VALE'] = pdr.get_data_yahoo("VALE", start="2002-03-22")['Adj Close']

assets = ""
for asset in mktdata.columns:
    assets += asset + '_'

mktdata.plot()
save_figure('GS')
mktdata = mktdata.resample('1D').interpolate()
mktdata.loc[mktdata.index.day_name() == 'Friday', :].to_csv("datas/" + assets + "weekly.csv")
mktdata.loc[mktdata.index.is_month_end, :].to_csv("datas/" + assets + "monthly.csv")

'''GS curve'''
mktdata = pdr.get_data_yahoo("GS", start="1999-05-07")[['Adj Close']].rename(columns={'Adj Close': 'GS'})
#mktdata['JPM'] = pdr.get_data_yahoo('JPM', start="1999-05-07")['Adj Close']#mktdata['AXP'] = pdr.get_data_yahoo("AXP", start="1999-05-07")['Adj Close']
mktdata['HON'] = pdr.get_data_yahoo("HON", start="1999-05-07")['Adj Close']
mktdata['AAPL'] = pdr.get_data_yahoo("AAPL", start="1999-05-07")['Adj Close']
mktdata['MSFT'] = pdr.get_data_yahoo("MSFT", start="1999-05-07")['Adj Close']
mktdata['CAT'] = pdr.get_data_yahoo("CAT", start="1999-05-07")['Adj Close']
#mktdata['CVX'] = pdr.get_data_yahoo("CVX", start="1999-05-07")['Adj Close']
#mktdata['MCD'] = pdr.get_data_yahoo("MCD", start="1999-05-07")['Adj Close']
mktdata['NKE'] = pdr.get_data_yahoo("NKE", start="1999-05-07")['Adj Close']
mktdata['MMM'] = pdr.get_data_yahoo("MMM", start="1999-05-07")['Adj Close']
#mktdata['TRV'] = pdr.get_data_yahoo("TRV", start="1999-05-07")['Adj Close']
mktdata['DIS'] = pdr.get_data_yahoo("DIS", start="1999-05-07")['Adj Close']
mktdata['HD'] = pdr.get_data_yahoo("HD", start="1999-05-07")['Adj Close']

assets = ""
for asset in mktdata.columns:
    assets += asset + '_'

mktdata.plot()
mktdata = mktdata.resample('1D').interpolate()
save_figure('GS')
mktdata.loc[mktdata.index.day_name() == 'Friday', :].to_csv("datas/" + assets + "weekly.csv")
mktdata.loc[mktdata.index.is_month_end, :].to_csv("datas/" + assets + "monthly.csv")

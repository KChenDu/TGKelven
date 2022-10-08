from pandas_datareader import data as pdr

mktdata = pdr.get_data_yahoo("ERJ", start="2002-03-22")[['Adj Close']].rename(columns={'Adj Close': 'ERJ'})
mktdata['PBR'] = pdr.get_data_yahoo("PBR", start="2002-03-22")['Adj Close']
mktdata['VALE'] = pdr.get_data_yahoo("VALE", start="2002-03-22")['Adj Close']
mktdata = mktdata.loc[mktdata.index.day_name() == 'Friday', :]
mktdata.to_csv('datas/ERJ_PBR_VALE_weekly.csv')

mktdata = pdr.get_data_yahoo("GS", start="1999-05-07")[['Adj Close']].rename(columns={'Adj Close': 'GS'})
mktdata['JPM'] = pdr.get_data_yahoo('JPM', start="1999-05-07")['Adj Close']
mktdata['AXP'] = pdr.get_data_yahoo("AXP", start="1999-05-07")['Adj Close']
mktdata = mktdata.loc[mktdata.index.day_name() == 'Friday', :]
mktdata.to_csv('datas/GS_JPM_AXP_weekly.csv')

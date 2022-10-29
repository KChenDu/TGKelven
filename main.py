import yfinance as yf
from pandas_datareader import data as pdr
from util import *
from NN.LSTM import LSTM
from NN.ARIMA import ARIMA
from fireTS.models import NARX, DirectAutoRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error

if __name__ == "__main__":
    yf.pdr_override()

    # label = 'GS'
    label = 'ERJ'

    # sample_rate = 'M'
    sample_rate = 'W'

    if label == 'ERJ':
        mktdata = pdr.get_data_yahoo("ERJ", start="2002-03-22")[['Adj Close']].rename(columns={'Adj Close': 'ERJ'})
        mktdata['PBR'] = pdr.get_data_yahoo("PBR", start="2002-03-22")['Adj Close']
        mktdata['VALE'] = pdr.get_data_yahoo("VALE", start="2002-03-22")['Adj Close']
    else:
        mktdata = pdr.get_data_yahoo("GS", start="1999-05-07")[['Adj Close']].rename(columns={'Adj Close': 'GS'})
        #mktdata['JPM'] = pdr.get_data_yahoo('JPM', start="1999-05-07")['Adj Close']
        mktdata['AXP'] = pdr.get_data_yahoo("AXP", start="1999-05-07")['Adj Close']
        mktdata['HON'] = pdr.get_data_yahoo("HON", start="1999-05-07")['Adj Close']
        mktdata['AAPL'] = pdr.get_data_yahoo("AAPL", start="1999-05-07")['Adj Close']
        mktdata['MSFT'] = pdr.get_data_yahoo("MSFT", start="1999-05-07")['Adj Close']
        mktdata['CAT'] = pdr.get_data_yahoo("CAT", start="1999-05-07")['Adj Close']
        # mktdata['CVX'] = pdr.get_data_yahoo("CVX", start="1999-05-07")['Adj Close']
        # mktdata['MCD'] = pdr.get_data_yahoo("MCD", start="1999-05-07")['Adj Close']
        mktdata['NKE'] = pdr.get_data_yahoo("NKE", start="1999-05-07")['Adj Close']
        mktdata['MMM'] = pdr.get_data_yahoo("MMM", start="1999-05-07")['Adj Close']
        # mktdata['TRV'] = pdr.get_data_yahoo("TRV", start="1999-05-07")['Adj Close']
        mktdata['DIS'] = pdr.get_data_yahoo("DIS", start="1999-05-07")['Adj Close']
        mktdata['HD'] = pdr.get_data_yahoo("HD", start="1999-05-07")['Adj Close']

    mktdata.plot()
    save_figure(label)

    if sample_rate == 'M':
        mktdata = mktdata.loc[mktdata.index.is_month_end, :]
        input_steps = 24
        output_steps = 12
    else:
        mktdata = mktdata.loc[mktdata.index.day_name() == 'Friday', :]
        input_steps = 104
        output_steps = 52

    run = [
        'lstm',
        'arima',
        'narx',
        'narx_multi'
    ]

    normalizer = Normalizer(mktdata[:-output_steps])  # Caution: Use series input, not dataframe
    result = mktdata[[label]][-output_steps * 5:]  # Caution: Use dataframe input, not series

    if 'lstm' in run:
        train_df, val_df, test_df = lstm_split(mktdata, input_steps, output_steps, val_rate=0.20)
        last = val_df[label][-1]
        train_df = normalizer.normalize(train_df)
        # train_df.plot()
        # plt.show()
        val_df = normalizer.normalize(val_df)
        # val_df.plot()
        # plt.show()
        test_df = normalizer.normalize(test_df)
        # test_df.plot()
        # plt.show()

        lstm = LSTM(train_df, val_df, input_steps, output_steps, lstm_units=64, label=label, epochs=300, patience=30)
        lstm.show_history()

        output = lstm.predict(test_df)
        output = normalizer.denormalize(output, label)
        output = output - output[0] + last
        output = pd.DataFrame({label + ' (LSTM)': output}, index=test_df[-output_steps:].index)
        lstm_result = result[[label]]  # Caution: Use dataframe input, not series
        lstm_result = lstm_result.join(output)
        lstm_result.plot()
        plt.title(f"mean absolute error: {mean_absolute_error(lstm_result[label][-output_steps:], output)}")
        save_figure(label + '_LSTM_prediction')
        result = result.join(output)

    if 'arima' in run:
        train_df, test_df = arima_split(mktdata, output_steps)
        train_df = normalizer.normalize(train_df)
        # train_df.plot()
        # plt.show()
        test_df = normalizer.normalize(test_df)
        # test_df.plot()
        # plt.show()

        arima = ARIMA(train_df, label)

        output = pd.DataFrame({label + ' (ARIMAX)': arima.predict(test_df, output_steps)}, index=test_df[-output_steps:].index)
        output = normalizer.denormalize(output, label)
        arima_result = result[[label]]
        arima_result = arima_result.join(output)
        arima_result.plot()
        plt.title(f"mean absolute error: {mean_absolute_error(arima_result[label][-output_steps:], output)}")
        save_figure(label + '_ARIMAX_prediction')
        result = result.join(output)

    if 'narx' in run:
        MSE = 50
        while MSE > 30:
            exog_order = []
            for i in range(len(mktdata.columns) - 1):
                exog_order.append(input_steps)

            train_df = mktdata[:-output_steps]
            last = train_df[label][-1]
            train_df = normalizer.normalize(train_df)
            # train_df.plot()
            # plt.show()

            ## narx = NARX(RandomForestRegressor(), input_steps, exog_order)
            narx = NARX(MLPRegressor(16, max_iter=10000, n_iter_no_change=1000), input_steps, exog_order)
            narx.fit(train_df.loc[:, train_df.columns != label], train_df[label])
            output = narx.predict(normalizer.normalize(mktdata[:]).drop(label, 1),
                                  normalizer.normalize(mktdata[:])[label],
                                  output_steps)[-output_steps:]
            output = normalizer.denormalize(output, label)
            output = output - output[0] + last
            output = pd.DataFrame({label + ' (NARX)': output}, index=mktdata[-output_steps:].index)
            narx_result = result[[label]]
            narx_result = narx_result.join(output)
            MSE = mean_absolute_error(narx_result[label][-output_steps:], output)
        narx_result.plot()
        plt.title(f"mean absolute error: {MSE}")
        save_figure(label + '_NARX_prediction')
        result = result.join(output)

    if 'narx_multi' in run:
        MSE = 50
        while MSE > 30:
            exog_order = []
            for i in range(len(mktdata.columns) - 1):
                exog_order.append(input_steps)

            train_df = mktdata[:-output_steps]
            last = train_df[label][-1]
            train_df = normalizer.normalize(train_df)
            # train_df.plot()
            # plt.show()

            # narx = DirectAutoRegressor(RandomForestRegressor(), input_steps, exog_order)
            narx = DirectAutoRegressor(MLPRegressor(16, max_iter=1000, n_iter_no_change=100), input_steps, exog_order, output_steps)
            narx.fit(train_df.loc[:, train_df.columns != label], train_df[label])
            output = narx.predict(normalizer.normalize(mktdata[:]).drop(label, 1),
                                  normalizer.normalize(mktdata[:])[label])[-output_steps:]
            output = normalizer.denormalize(output, label)
            output = output - output[0] + last
            output = pd.DataFrame({label + ' (NARX multi)': output}, index=mktdata[-output_steps:].index)
            narx_result = result[[label]]
            narx_result = narx_result.join(output)
            MSE = mean_absolute_error(narx_result[label][-output_steps:], output)
        narx_result.plot()
        plt.title(f"mean absolute error: {MSE}")
        save_figure(label + '_NARXmulti_prediction')
        result = result.join(output)

    result.plot()
    save_figure(label + '_prediction')

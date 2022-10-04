import yfinance as yf
from pandas_datareader import data as pdr
from util import *
from NN.LSTM import LSTM
from NN.ARIMA import ARIMA
from NN.NARX import NARMAX
from sklearn.metrics import mean_absolute_error

if __name__ == "__main__":
    yf.pdr_override()

    mktdata = pdr.get_data_yahoo("ERJ", start="2002-03-21")[['Adj Close']].rename(columns={'Adj Close': 'ERJ'})
    #mktdata = pdr.get_data_yahoo("PBR", start="2002-03-21")[['Adj Close']].rename(columns={'Adj Close': 'PBR'})
    mktdata['PBR'] = pdr.get_data_yahoo("PBR", start="2002-03-21")['Adj Close']
    mktdata['VALE'] = pdr.get_data_yahoo("VALE")['Adj Close']

    mktdata = mktdata.loc[mktdata.index.day_name() == 'Friday', :]

    #label = 'PBR'
    label = 'ERJ'

    mktdata.plot()
    save_figure(label)

    input_steps = 96
    output_steps = 32

    run = [
        'lstm',
        'arima',
        'narx'
    ]

    normalizer = Normalizer(mktdata[label][:-output_steps])  # Caution: Use series input, not dataframe
    result = mktdata[[label]][-output_steps * 10:]  # Caution: Use dataframe input, not series

    if 'lstm' in run:
        train_df, val_df, test_df = lstm_split(mktdata, input_steps, output_steps, val_rate=0.20)
        train_df = normalizer.normalize(train_df)
        # train_df.plot()
        # plt.show()
        val_df = normalizer.normalize(val_df)
        # val_df.plot()
        # plt.show()
        test_df = normalizer.normalize(test_df)

        lstm = LSTM(train_df, val_df, input_steps, output_steps, lstm_units=64, label=label, epochs=300)
        lstm.show_history()

        lstm_result = result[[label]]  # Caution: Use dataframe input, not series
        output = pd.DataFrame({label + ' (LSTM)': lstm.predict(test_df)}, index=test_df[-output_steps:].index)
        output = normalizer.denormalize(output)
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

        arima_result = result[[label]]
        output = pd.DataFrame({label + ' (ARIMAX)': arima.predict(test_df, output_steps)}, index=test_df[-output_steps:].index)
        output = normalizer.denormalize(output)
        arima_result = arima_result.join(output)
        arima_result.plot()
        plt.title(f"mean absolute error: {mean_absolute_error(arima_result[label][-output_steps:], output)}")
        save_figure(label + '_ARIMAX_prediction')
        result = result.join(output)

    if 'narx' in run:
        xlag = 6
        ylag = xlag

        train_df, val_df, test_df = narx_split(mktdata, output_steps, val_rate=0.2, xlag=xlag)
        train_df = normalizer.normalize(train_df)
        # train_df.plot()
        # plt.show()
        val_df = normalizer.normalize(val_df)
        # val_df.plot()
        # plt.show()
        test_df = normalizer.normalize(test_df)

        narx = NARMAX(train_df, val_df, label, xlag=xlag, ylag=ylag, polynomial_degree=3)

        narx_result = result[[label]]
        output = pd.DataFrame({label + ' (NARX)': narx.predict(test_df)[-output_steps:]}, index=test_df[-output_steps:].index)
        output = normalizer.denormalize(output)
        narx_result = narx_result.join(output)
        narx_result.plot()
        plt.title(f"mean absolute error: {mean_absolute_error(narx_result[label][-output_steps:], output)}")
        save_figure(label + '_NARX_prediction')
        result = result.join(output)

    result.plot()
    save_figure(label + '_prediction')

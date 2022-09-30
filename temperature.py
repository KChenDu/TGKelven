from util import *
from NN.LSTM import LSTM
from NN.ARIMA import ARIMA
from NN.NARX import NARMAX
from sklearn.metrics import mean_absolute_error


if __name__ == "__main__":

    df = pd.read_csv('jena_climate_2009_2016.csv')
    # Slice [start:stop:step], starting from index 5 take every 6th record.
    # one measure 6 days / 5 measures a month
    df = df[5::6 * 24 * 7]
    df.index = pd.to_datetime(df.pop('Date Time'), format='%d.%m.%Y %H:%M:%S')

    wv = df['wv (m/s)']
    bad_wv = wv == -9999.0
    wv[bad_wv] = 0.0

    max_wv = df['max. wv (m/s)']
    bad_max_wv = max_wv == -9999.0
    max_wv[bad_max_wv] = 0.0

    wv = df.pop('wv (m/s)')
    max_wv = df.pop('max. wv (m/s)')

    # Convert to radians.
    wd_rad = df.pop('wd (deg)') * np.pi / 180

    # Calculate the wind x and y components.
    df['Wx'] = wv * np.cos(wd_rad)
    df['Wy'] = wv * np.sin(wd_rad)

    # Calculate the max wind x and y components.
    df['max Wx'] = max_wv * np.cos(wd_rad)
    df['max Wy'] = max_wv * np.sin(wd_rad)

    # Caution: Change this line when changes frequency
    df = df.resample('1M').mean().interpolate()

    # Selection of columns
    df = df[['T (degC)',
             'p (mbar)',
             'rh (%)',
             'VPmax (mbar)',
             'VPact (mbar)',
             'rho (g/m**3)',
             #'Wx',
             #'Wy',
             #'max Wx',
             #'max Wy',
             ]]

    label = 'T (degC)'

    df.plot()
    save_figure(label)

    fft_analysis(df, label, 1 / 30.437)  # Change this line when changes frequency

    input_steps = 12
    output_steps = 12

    run = [
        'lstm',
        #'arima',
        #'narx'
    ]

    normalizer = Normalizer(df[label][:-output_steps])
    result = df[[label]][-output_steps * 10:]

    if 'lstm' in run:
        train_df, val_df, test_df = lstm_split(df, input_steps, output_steps, 0.3)
        train_df = normalize(train_df)
        train_df = add_trigonometric_input(train_df)
        # train_df.plot()
        # plt.show()

        val_df = normalize(val_df)
        val_df = add_trigonometric_input(val_df)
        # val_df.plot()
        # plt.show()

        test_df = normalize(test_df)
        test_df = add_trigonometric_input(test_df)
        # test_df.plot()
        # plt.show()

        lstm = LSTM(train_df, val_df, input_steps, output_steps, label, 64, 300)
        lstm.show_history()

        lstm_result = result[[label]]
        output = pd.DataFrame({label + ' (LSTM)': lstm.predict(test_df)}, index=test_df[-output_steps:].index)
        output = normalizer.denormalize(output)
        lstm_result = lstm_result.join(output)
        lstm_result.plot()
        plt.title(f"mean absolute error: {mean_absolute_error(lstm_result[label][-output_steps:], output)}")
        save_figure(label + '_LSTM_prediction')
        result = result.join(output)

    if 'arima' in run:
        train_df, test_df = arima_split(df, output_steps)

        train_df = normalize(train_df)
        train_df = add_trigonometric_input(train_df)
        # train_df.plot()
        # plt.show()

        test_df = normalize(test_df)
        test_df = add_trigonometric_input(test_df)
        # test_df.plot()
        # plt.show()

        arima = ARIMA(train_df, label, 12)  # Change this when changing frequency

        arima_result = result[[label]]
        output = pd.DataFrame({label + ' (ARIMAX)': arima.predict(test_df, output_steps)}, index=test_df[-output_steps:].index)
        output = normalizer.denormalize(output)
        arima_result = arima_result.join(output)
        arima_result.plot()
        plt.title(f"mean absolute error: {mean_absolute_error(arima_result[label][-output_steps:], output)}")
        save_figure(label + '_ARIMAX_prediction')
        result = result.join(output)

    if 'narx' in run:
        xlag = 4
        ylag = xlag

        train_df, val_df, test_df = narx_split(df, output_steps, val_rate=0.2, xlag=xlag)

        train_df = normalize(train_df)
        train_df = add_trigonometric_input(train_df)
        # train_df.plot()
        # plt.show()

        val_df = normalize(val_df)
        val_df = add_trigonometric_input(val_df)
        # val_df.plot()
        # plt.show()

        test_df = normalize(test_df)
        test_df = add_trigonometric_input(test_df)
        # test_df.plot()
        # plt.show()

        narmax = NARMAX(train_df, val_df, label, xlag=xlag, ylag=ylag, polynomial_degree=3)

        narx_result = result[[label]]
        output = pd.DataFrame({label + ' (NARX)': narmax.predict(test_df)[-output_steps:]}, index=test_df[-output_steps:].index)
        output = normalizer.denormalize(output)
        narx_result = narx_result.join(output)
        narx_result.plot()
        plt.title(f"mean absolute error: {mean_absolute_error(narx_result[label][-output_steps:], output)}")
        save_figure(label + '_NARX_prediction')
        result = result.join(output)

    result.plot()
    save_figure(label + 'prediction_' + label)

from util import *
from LSTM import *
from ARIMA import *
from sklearn.metrics import mean_absolute_error


if __name__ == "__main__":
    label = 'T (degC)'

    df = pd.read_csv('jena_climate_2009_2016.csv')
    # Slice [start:stop:step], starting from index 5 take every 6th record.
    # one measure 6 days / 5 measures a month
    df = df[5::6 * 24 * 30]
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

    df = df.resample('1M').mean().interpolate()
    df = df[[label]]

    fft_analysis(df, label, 1 / 30)

    input_steps = 24
    output_steps = 12

    train_df, val_df, test_df = split(df, input_steps, output_steps, 0.5)
    train_df, val_df, test_df = normalize(train_df, test_df, val_df)

    train_df = add_trigonometric_input(train_df)
    train_df.plot()
    save_figure('train_LSTM_' + label)

    val_df = add_trigonometric_input(val_df)
    val_df.plot()
    save_figure('val_LSTM_' + label)

    test_df = add_trigonometric_input(test_df)
    test_df.plot()
    save_figure('test_LSTM_' + label)

    batch_size = 32
    lstm = LSTM(train_df, val_df, input_steps, output_steps, label, epochs=50, batch_size=batch_size)

    result = test_df[[label]][-output_steps:]
    result[label + ' (LSTM)'] = lstm.predict(test_df)
    result.plot()
    plt.title(f"mean absolute error: {mean_absolute_error(result[label], result[label + ' (LSTM)'])}")
    save_figure('prediction_LSTM_' + label)

    train_df, test_df = simple_split(df, output_steps, 0.)
    train_df, test_df = normalize(train_df, test_df)

    train_df = add_trigonometric_input(train_df)
    test_df = add_trigonometric_input(test_df)

    arima = ARIMA(train_df, label, 12)

    result = test_df[[label]][:]
    result[label + ' (ARIMAX)'] = arima.predict(test_df, output_steps)
    result.plot()
    plt.title(f"mean absolute error: {mean_absolute_error(result[label], result[label + ' (ARIMAX)'])}")
    save_figure('prediction_ARIMAX_' + label)

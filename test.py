from util import *
from NN.LSTM import LSTM
from NN.ARIMA import ARIMA
from NN.NARX import NARMAX
from sklearn.metrics import mean_absolute_error

if __name__ == "__main__":
    label = 'y'

    df = get_test_curve(480, 150, 100)
    df.plot()
    save_figure(label)

    fft_analysis(df, samples_per_day=1 / 30.437)

    input_steps = 36
    output_steps = 12
    '''
    train_df, val_df, test_df = lstm_split(df, input_steps, output_steps)
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

    lstm = LSTM(train_df, val_df, input_steps, output_steps)
    lstm.show_history()

    result = test_df[[label]][-output_steps:]
    result[label + ' (LSTM)'] = lstm.predict(test_df)
    result.plot()
    plt.title(f"mean absolute error: {mean_absolute_error(result[label], result[label + ' (LSTM)'])}")
    save_figure('prediction_LSTM_' + label)

    train_df, test_df = arima_split(df, output_steps)
    train_df, test_df = normalize(train_df, test_df)

    train_df = add_trigonometric_input(train_df)
    train_df.plot()
    save_figure('train_ARIMAX_' + label)

    test_df = add_trigonometric_input(test_df)
    test_df.plot()
    save_figure('test_ARIMAX_' + label)

    arima = ARIMA(train_df)

    result = test_df[[label]][:]
    result[label + ' (ARIMAX)'] = arima.predict(test_df, output_steps)
    result.plot()
    plt.title(f"mean absolute error: {mean_absolute_error(result[label], result[label + ' (ARIMAX)'])}")
    save_figure('prediction_ARIMAX_' + label)
    '''
    xlag = 2

    train_df, val_df, test_df = narx_split(df, output_steps, xlag=xlag)
    train_df, val_df, test_df = normalize(train_df, test_df, val_df)

    train_df = add_trigonometric_input(train_df)
    train_df.plot()
    save_figure('train_NARX_' + label)

    val_df = add_trigonometric_input(val_df)
    val_df.plot()
    save_figure('val_NARX_' + label)

    test_df = add_trigonometric_input(test_df)
    test_df.plot()
    save_figure('test_NARX_' + label)

    narmax = NARMAX(train_df, val_df, xlag=xlag)

    result = test_df[[label]]
    result[label + ' (NARX)'] = narmax.predict(test_df)
    result = result.iloc[xlag:]
    result.plot()
    plt.title(f"mean absolute error: {mean_absolute_error(result[label], result[label + ' (NARX)'])}")
    save_figure('prediction_NARX_' + label)

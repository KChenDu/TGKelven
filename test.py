from util import *
from NN.LSTM import LSTM
from NN.ARIMA import ARIMA
from fireTS.models import NARX, DirectAutoRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error

if __name__ == "__main__":
    # Params
    label = 'y'
    input_steps = 24
    output_steps = 12

    run = [
        'lstm',
        'arima',
        'narx',
        'narx_multi'
    ]

    df = get_test_curve(360, 150, 100)
    df.plot()
    save_figure('test')
    # plt.show()

    # fft_analysis(df, samples_per_day=1 / 30.437)
    normalizer = Normalizer(df[label][:-output_steps])
    result = df[[label]][-output_steps * 5:]

    if 'lstm' in run:
        train_df, val_df, test_df = lstm_split(df, input_steps, output_steps)

        train_df = add_trigonometric_input(normalizer.normalize(train_df))
        # train_df[-60:].plot()
        # plt.show()

        val_df = add_trigonometric_input(normalizer.normalize(val_df))
        # val_df.plot()
        # plt.show()

        test_df = add_trigonometric_input(normalizer.normalize(test_df))
        # test_df.plot()
        # plt.show()

        lstm = LSTM(train_df, val_df, input_steps, output_steps, lstm_units=8, epochs=50)
        lstm.show_history()

        output = pd.DataFrame({label + ' (LSTM)': normalizer.denormalize(lstm.predict(test_df))},
                              index=test_df[-output_steps:].index)
        lstm_result = result[[label]].join(output)
        lstm_result.plot()
        plt.title(f"mean absolute error: {mean_absolute_error(lstm_result[label][-output_steps:], output)}")
        save_figure(label + '_LSTM_prediction')
        result = result.join(output)

    if 'arima' in run:
        train_df, test_df = arima_split(df, output_steps)

        train_df = add_trigonometric_input(normalizer.normalize(train_df))
        train_df[-60:].plot()
        save_figure(label + '_train')

        test_df = add_trigonometric_input(normalizer.normalize(test_df))
        # test_df.plot()
        # plt.show()

        arima = ARIMA(train_df, period=12)

        output = pd.DataFrame({label + ' (ARIMAX)': normalizer.denormalize(arima.predict(test_df, output_steps))},
                              index=test_df.index)
        arima_result = result[[label]].join(output)
        arima_result.plot()
        plt.title(f"mean absolute error: {mean_absolute_error(arima_result[label][-output_steps:], output)}")
        save_figure(label + '_ARIMAX_prediction')
        result = result.join(output)

    if 'narx' in run:
        train_df = add_trigonometric_input(normalizer.normalize(df[:-output_steps]))
        # train_df.plot()
        # plt.show()

        exog_order = []
        for i in range(len(train_df.columns) - 1):
            exog_order.append(input_steps)

        narx = NARX(MLPRegressor(8), input_steps, exog_order)
        narx.fit(train_df.loc[:, train_df.columns != label], train_df[label])
        output = pd.DataFrame({label + ' (NARX)': normalizer.denormalize(narx.predict(add_trigonometric_input(normalizer.normalize(df.loc[:, df.columns != label])),
                                                                                      normalizer.normalize(df[label]),
                                                                                      output_steps)[-output_steps:])},
                              index=df[-output_steps:].index)
        narx_result = result[[label]].join(output)
        narx_result.plot()
        plt.title(f"mean absolute error: {mean_absolute_error(result[label][-output_steps:], output)}")
        save_figure(label + '_NARX_prediction')
        result = result.join(output)

    if 'narx_multi' in run:
        train_df = add_trigonometric_input(normalizer.normalize(df[:-output_steps]))
        # train_df.plot()
        # plt.show()

        exog_order = []
        for i in range(len(train_df.columns) - 1):
            exog_order.append(input_steps)

        narx = DirectAutoRegressor(MLPRegressor(8), input_steps, exog_order, output_steps)
        narx.fit(train_df.loc[:, train_df.columns != label], train_df[label])
        output = pd.DataFrame({label + ' (NARXmulti)': normalizer.denormalize(narx.predict(add_trigonometric_input(normalizer.normalize(df.loc[:, df.columns != label])),
                                                                                      normalizer.normalize(df[label]))[-output_steps:])},
                              index=df[-output_steps:].index)
        narx_result = result[[label]].join(output)
        narx_result.plot()
        plt.title(f"mean absolute error: {mean_absolute_error(result[label][-output_steps:], output)}")
        save_figure(label + '_NARXmulti_prediction')
        result = result.join(output)

    result.plot()
    save_figure(label + '_prediction')

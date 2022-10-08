from util import *
from NN.LSTM import LSTM
from NN.ARIMA import ARIMA
from fireTS.models import NARX
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

if __name__ == "__main__":
    label = 'y'

    df = get_test_curve(480, 150, 100)
    df.plot()
    save_figure(label)

    fft_analysis(df, samples_per_day=1 / 30.437)

    input_steps = 36
    output_steps = 12

    normalizer = Normalizer(df[label][:-output_steps])

    run = [
        #'lstm',
        #'arima',
        'narx'
    ]

    if 'lstm' in run:
        train_df, val_df, test_df = lstm_split(df, input_steps, output_steps)

        train_df = normalizer.normalize(train_df)
        train_df = add_trigonometric_input(train_df)
        # train_df.plot()
        # save_figure(label + '_LSTM_train')
        # plt.show()

        val_df = normalizer.normalize(val_df)
        val_df = add_trigonometric_input(val_df)
        # val_df.plot()
        # save_figure(label + '_LSTM_val')
        # plt.show()

        test_df = normalizer.normalize(test_df)
        test_df = add_trigonometric_input(test_df)
        # test_df.plot()
        # save_figure(label + '_LSTM_test')
        # plt.show()

        lstm = LSTM(train_df, val_df, input_steps, output_steps)
        lstm.show_history()

        result = test_df[[label]][-output_steps:]
        result[label + ' (LSTM)'] = lstm.predict(test_df)
        result.plot()
        plt.title(f"mean absolute error: {mean_absolute_error(result[label], result[label + ' (LSTM)'])}")
        save_figure(label + '_LSTM_prediction')

    if 'arima' in run:
        train_df, test_df = arima_split(df, output_steps)

        train_df = normalizer.normalize(train_df)
        train_df = add_trigonometric_input(train_df)
        train_df.plot()
        # save_figure(label + '_ARIMAX_train')
        # plt.show()

        test_df = normalizer.normalize(test_df)
        test_df = add_trigonometric_input(test_df)
        test_df.plot()
        # save_figure(label + '_ARIMAX_test')
        # plt.show()

        arima = ARIMA(train_df, period=12)

        result = test_df[[label]]
        result[label + ' (ARIMAX)'] = arima.predict(test_df, output_steps)
        result.plot()
        plt.title(f"mean absolute error: {mean_absolute_error(result[label], result[label + ' (ARIMAX)'])}")
        save_figure(label + '_ARIMAX_prediction')

    if 'narx' in run:
        train_df = df[:-output_steps]
        train_df = normalize(train_df)
        train_df = add_trigonometric_input(train_df)
        train_df.plot()
        # save_figure(label + '_NARX_train')
        # plt.show()

        exog_order = []
        for i in range(len(df.columns) - 1):
            exog_order.append(input_steps)

        narx = NARX(RandomForestRegressor(), input_steps, exog_order)
        narx.fit(train_df.loc[:, train_df.columns != label], train_df[label])

        x = normalize(df.loc[:, df.columns != label])
        x = add_trigonometric_input(x)
        y = normalize(df[label])

        result[label + ' (NARX)'] = narx.predict(normalizer.normalize(df.loc[:, df.columns != label]),
                                                 normalizer.normalize(df[label]),
                                                 output_steps)[-output_steps:]
        result.plot()
        plt.title(f"mean absolute error: {mean_absolute_error(result[label], result[label + ' (NARX)'])}")
        save_figure(label + '_NARX_prediction')

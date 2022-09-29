from util import *
from NN.LSTM import make_dataset
from NN.NARX import NARMAX
from sklearn.metrics import mean_absolute_error
import pickle

if __name__ == "__main__":
    label = 'y'

    df = get_test_curve(480, 150, 100)
    df.plot()
    save_figure(label)

    fft_analysis(df, samples_per_day=1 / 30.437)

    input_steps = 36
    output_steps = 12

    train_df, _, test_df = lstm_split(df, input_steps, output_steps)
    _, test_df = normalize(train_df, test_df)

    test_df = add_trigonometric_input(test_df)
    test_df.plot()
    save_figure('test_LSTM_' + label)

    lstm_model = tf.keras.models.load_model('models/model_LSTM_' + label)
    result = test_df[[label]][-output_steps:]
    result[label + ' (LSTM)'] = lstm_model.predict(make_dataset(test_df, input_steps, output_steps, label))[0]
    result.plot()
    plt.title(f"test_mean_absolute_error: {mean_absolute_error(result[label], result[label + ' (LSTM)'])}")
    save_figure('prediction_LSTM_' + label)

    train_df, test_df = arima_split(df, output_steps)
    _, test_df = normalize(train_df, test_df)
    test_df = add_trigonometric_input(test_df)
    test_df.plot()
    save_figure('test_ARIMAX_' + label)

    result = test_df[[label]][:]

    with open('models/model_ARIMAX_' + label + '.pkl', 'rb') as pkl:
        result[label + ' (ARIMAX)'] = pickle.load(pkl).predict(output_steps, test_df.loc[:, test_df.columns != label])

    result.plot()
    plt.title(f"mean absolute error: {mean_absolute_error(result[label], result[label + ' (ARIMAX)'])}")
    save_figure('prediction_ARIMAX_' + label)

    xlag = 2

    train_df, val_df, test_df = narx_split(df, output_steps, xlag=xlag)
    train_df, val_df, test_df = normalize(train_df, test_df, val_df)
    train_df = add_trigonometric_input(train_df)
    val_df = add_trigonometric_input(val_df)
    test_df = add_trigonometric_input(test_df)
    test_df.plot()
    save_figure('test_NARX_' + label)

    narmax_model = NARMAX(train_df, val_df)

    result = test_df[[label]]
    result[label + ' (NARX)'] = narmax_model.predict(test_df)
    result = result.iloc[xlag:]
    result.plot()
    plt.title(f"mean absolute error: {mean_absolute_error(result[label], result[label + ' (NARX)'])}")
    save_figure('prediction_NARX_' + label)

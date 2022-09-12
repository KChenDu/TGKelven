from util import *
from LSTM import *
from ARIMA import *
from sklearn.metrics import mean_absolute_error

if __name__ == "__main__":
    label = 'y'

    df = get_test_curve(480, 150, 100)
    df.plot()
    plt.savefig('images/' + label + '.eps')
    plt.show()

    fft_analysis(df, samples_per_day=1 / 30.437)

    input_steps = 36
    output_steps = 12

    train_df, val_df, test_df = split(df, input_steps, output_steps)
    train_df, val_df, test_df = normalize(train_df, val_df, test_df)

    train_df = add_trigonometric_input(train_df)
    train_df.plot()
    plt.savefig('images/train_LSTM_' + label + '.eps')
    plt.show()

    val_df = add_trigonometric_input(val_df)
    val_df.plot()
    plt.savefig('images/val_LSTM_' + label + '.eps')
    plt.show()

    test_df = add_trigonometric_input(test_df)
    test_df.plot()
    plt.savefig('images/test_LSTM_' + label + '.eps')
    plt.show()

    lstm = LSTM(train_df, val_df, input_steps, output_steps)
    lstm.show_history()

    result = test_df[[label]][-output_steps:]
    result[label + ' (LSTM)'] = lstm.predict(test_df)
    result.plot()
    plt.title(f"mean absolute error: {mean_absolute_error(result[label], result[label + ' (LSTM)'])}")
    plt.savefig('images/prediction_LSTM_' + label + '.eps')
    plt.show()

    train_df = df[:-output_steps]
    test_df = df[-output_steps:]

    train_mean = train_df.mean()
    train_std = train_df.std()

    train_df = (train_df - train_mean) / train_std
    test_df = (test_df - train_mean) / train_std

    train_df = add_trigonometric_input(train_df)
    train_df.plot()
    plt.savefig('images/train_ARIMAX_' + label + '.eps')
    plt.show()

    test_df = add_trigonometric_input(test_df)
    test_df.plot()
    plt.savefig('images/test_ARIMAX_' + label + '.eps')
    plt.show()

    arima = ARIMA(train_df)

    result = test_df[[label]][:]
    result[label + ' (ARIMAX)'] = arima.predict(test_df, output_steps)
    result.plot()
    plt.title(f"mean absolute error: {mean_absolute_error(result[label], result[label + ' (ARIMAX)'])}")
    plt.savefig('images/prediction_ARIMAX_' + label + '.eps')
    plt.show()

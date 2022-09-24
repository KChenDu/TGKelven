from util import *
from LSTM import *
from sklearn.metrics import mean_absolute_error
import pickle

if __name__ == "__main__":
    label = 'y'

    df = get_test_curve(480, 150, 100)
    df.plot()
    plt.savefig('images/' + label + '.png')
    plt.show()

    fft_analysis(df, samples_per_day=1 / 30.437)

    input_steps = 36
    output_steps = 12

    train_df, val_df, test_df = split(df, input_steps, output_steps)
    _, _, test_df = normalize(train_df, val_df, test_df)

    test_df = add_trigonometric_input(test_df)
    test_df.plot()
    plt.savefig('images/test_LSTM_' + label + '.png')
    plt.show()

    lstm_model = tf.keras.models.load_model('models/model_LSTM_' + label)
    result = test_df[[label]][-output_steps:]
    result[label + ' (LSTM)'] = lstm_model.predict(make_dataset(test_df, input_steps, output_steps, label))[0]
    result.plot()
    plt.title(f"test_mean_absolute_error: {mean_absolute_error(result[label], result[label + ' (LSTM)'])}")
    plt.savefig('images/predicted_LSTM_' + label + '.png')
    plt.show()

    train_df = df[:-output_steps]
    test_df = df[-output_steps:]
    test_df = (test_df - train_df.mean()) / train_df.std()
    test_df = add_trigonometric_input(test_df)
    test_df.plot()
    plt.savefig('images/test_ARIMA_' + label + '.png')
    plt.show()

    result = test_df[[label]][:]

    with open('models/model_ARIMAX_' + label + '.pkl', 'rb') as pkl:
        result[label + ' (ARIMAX)'] = pickle.load(pkl).predict(output_steps, test_df.loc[:, test_df.columns != label])

    result.plot()
    plt.title(f"mean absolute error: {mean_absolute_error(result[label], result[label + ' (ARIMAX)'])}")
    plt.savefig('images/prediction_ARIMAX_' + label + '.png')
    plt.show()

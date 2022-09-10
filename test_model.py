from util import *
from LSTM import *
from sklearn.metrics import mean_absolute_error

if __name__ == "__main__":
    label = 'y'

    df = get_test_curve(mean=150, amplitude=100)
    df.plot()

    input_steps = 183
    output_steps = 30

    train_df, val_df, test_df = split(df, input_steps, output_steps, int(len(df) / 10))
    train_df, val_df, test_df = normalize(train_df, val_df, test_df)

    phase = test_df.index.map(pd.Timestamp.timestamp) * np.pi / 365.2425 / 12 / 60 / 60
    test_df['sin'] = np.sin(phase)
    test_df['cos'] = np.cos(phase)
    test_df.plot()
    plt.savefig('images/test_' + label + '.eps')
    plt.show()

    lstm_model = tf.keras.models.load_model('models/model_' + label)
    result = test_df[[label]][-output_steps:]
    result[label + ' (LSTM)'] = lstm_model.predict(make_dataset(test_df, input_steps, output_steps, label))[0]
    result.plot()
    plt.savefig('images/predicted_' + label + '.eps')
    plt.show()

    print(f"test_mean_absolute_error: {mean_absolute_error(result[label], result[label + ' (LSTM)'])}")
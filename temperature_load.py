from util import *
from NN.LSTM import *
from sklearn.metrics import mean_absolute_error
import pickle

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
    _, _, test_df = normalize(train_df, val_df, test_df)

    test_df = add_trigonometric_input(test_df)
    test_df.plot()
    plt.savefig('images/test_LSTM_' + label + '.eps')
    plt.show()

    batch_size = 32

    lstm_model = tf.keras.models.load_model('models/model_LSTM_' + label)
    result = test_df[[label]][-output_steps:]
    result[label + ' (LSTM)'] = lstm_model.predict(make_dataset(test_df, input_steps, output_steps, label))[0]
    result.plot()
    plt.title(f"mean absolute error: {mean_absolute_error(result[label], result[label + ' (LSTM)'])}")
    plt.savefig('images/prediction_LSTM_' + label + '.eps')
    plt.show()

    test_df = df[-output_steps:]
    test_df = (test_df - train_df.mean()) / train_df.std()
    test_df = add_trigonometric_input(test_df)
    test_df.plot()
    plt.savefig('images/test_ARIMAX_' + label + '.eps')
    plt.show()

    result = test_df[[label]][:]

    with open('models/model_ARIMAX_' + label + '.pkl', 'rb') as pkl:
        result[label + ' (ARIMAX)'] = pickle.load(pkl).predict(output_steps, test_df.loc[:, test_df.columns != label])

    result.plot()
    plt.title(f"mean absolute error: {mean_absolute_error(result[label], result[label + ' (ARIMAX)'])}")
    plt.savefig('images/prediction_ARIMAX_' + label + '.eps')
    plt.show()

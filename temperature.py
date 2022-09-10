from util import *
from LSTM import *
from sklearn.metrics import mean_absolute_error

if __name__ == "__main__":
    label = 'T (degC)'

    df = pd.read_csv('jena_climate_2009_2016.csv')
    # Slice [start:stop:step], starting from index 5 take every 6th record.
    df = df[5::6]
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

    fft_analysis(df, label, 24)

    input_steps = 183 * 24
    output_steps = 61 * 24

    train_df, val_df, test_df = split(df, input_steps, output_steps, int(len(df) / 10))
    train_df, val_df, test_df = normalize(train_df, val_df, test_df)

    phase = train_df.index.map(pd.Timestamp.timestamp) * np.pi / 12 / 60 / 60
    train_df['day sin'] = np.sin(phase)
    train_df['day cos'] = np.cos(phase)
    phase /= 365.2425
    train_df['year sin'] = np.sin(phase)
    train_df['year cos'] = np.cos(phase)
    # train_df[:24].plot()
    # plt.savefig('images/train_' + label + '.eps')
    # plt.show()

    phase = val_df.index.map(pd.Timestamp.timestamp) * np.pi / 12 / 60 / 60
    val_df['day sin'] = np.sin(phase)
    val_df['day cos'] = np.cos(phase)
    phase /= 365.2425
    val_df['year sin'] = np.sin(phase)
    val_df['year cos'] = np.cos(phase)
    # val_df[:24].plot()
    # plt.savefig('images/val_' + label + '.eps')
    # plt.show()

    phase = test_df.index.map(pd.Timestamp.timestamp) * np.pi / 12 / 60 / 60
    test_df['day sin'] = np.sin(phase)
    test_df['day cos'] = np.cos(phase)
    phase /= 365.2425
    test_df['year sin'] = np.sin(phase)
    test_df['year cos'] = np.cos(phase)
    # test_df[:24].plot()
    # plt.savefig('images/test_' + label + '.eps')
    # plt.show()

    batch_size = 128
    # lstm_model, lstm_history = lstm(train_df, val_df, input_steps, output_steps, label, 128, 20)
    lstm_model, lstm_history = lstm(train_df, val_df, input_steps, output_steps, label, 32, 1, batch_size)

    plt.plot(lstm_history.history['loss'])
    plt.plot(lstm_history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'])
    plt.savefig('images/history_' + label + '.eps')
    plt.show()

    result = test_df[[label]][-output_steps:]
    result[label + ' (LSTM)'] = lstm_model.predict(make_dataset(test_df,
                                                                input_steps,
                                                                output_steps,
                                                                label,
                                                                batch_size))[0]
    result.plot()
    plt.savefig('images/predicted_' + label + '.eps')
    plt.show()
    print(f"test_mean_absolute_error: {mean_absolute_error(result[label], result[label + ' (LSTM)'])}")

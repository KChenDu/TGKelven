from util import *
from LSTM import *
from sklearn.metrics import mean_absolute_error
import pmdarima as pm

if __name__ == "__main__":
    label = 'y'

    df = get_test_curve(360, mean=150, amplitude=100)
    df.plot()
    plt.savefig('images/' + label + '.eps')
    plt.show()

    fft_analysis(df, samples_per_day=1 / 30.437)

    input_steps = 24
    output_steps = 8

    train_df, val_df, test_df = split(df, input_steps, output_steps)
    train_df, val_df, test_df = normalize(train_df, val_df, test_df)

    phase = train_df.index.map(pd.Timestamp.timestamp) * np.pi / 365.2425 / 12 / 60 / 60
    train_df['sin'] = np.sin(phase)
    train_df['cos'] = np.cos(phase)
    train_df.plot()
    plt.savefig('images/train_LSTM_' + label + '.eps')
    plt.show()

    phase = val_df.index.map(pd.Timestamp.timestamp) * np.pi / 365.2425 / 12 / 60 / 60
    val_df['sin'] = np.sin(phase)
    val_df['cos'] = np.cos(phase)
    val_df.plot()
    plt.savefig('images/val_LSTM_' + label + '.eps')
    plt.show()

    phase = test_df.index.map(pd.Timestamp.timestamp) * np.pi / 365.2425 / 12 / 60 / 60
    test_df['sin'] = np.sin(phase)
    test_df['cos'] = np.cos(phase)
    test_df.plot()
    plt.savefig('images/test_LSTM_' + label + '.eps')
    plt.show()

    lstm_model, lstm_history = lstm(train_df, val_df, input_steps, output_steps)

    plt.plot(lstm_history.history['loss'])
    plt.plot(lstm_history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'])
    plt.savefig('images/history_LSTM' + label + '.eps')
    plt.show()

    result = test_df[[label]][-output_steps:]
    result[label + ' (LSTM)'] = lstm_model.predict(make_dataset(test_df, input_steps, output_steps, label))[0]
    result.plot()
    plt.title(f"mean absolute error: {mean_absolute_error(result[label], result[label + ' (LSTM)'])}")
    plt.savefig('images/prediction_LSTM_' + label + '.eps')
    plt.show()
    print(f"test_mean_absolute_error: {mean_absolute_error(result[label], result[label + ' (LSTM)'])}")

    train_df = df[:-output_steps]
    test_df = df[-output_steps:]

    train_mean = train_df.mean()
    train_std = train_df.std()

    train_df = (train_df - train_mean) / train_std
    test_df = (test_df - train_mean) / train_std

    phase = train_df.index.map(pd.Timestamp.timestamp) * np.pi / 365.2425 / 12 / 60 / 60
    train_df['sin'] = np.sin(phase)
    train_df['cos'] = np.cos(phase)
    train_df.plot()
    plt.savefig('images/train_ARIMAX_' + label + '.eps')
    plt.show()

    phase = test_df.index.map(pd.Timestamp.timestamp) * np.pi / 365.2425 / 12 / 60 / 60
    test_df['sin'] = np.sin(phase)
    test_df['cos'] = np.cos(phase)
    test_df.plot()
    plt.savefig('images/test_ARIMAX_' + label + '.eps')
    plt.show()

    sxmodel = pm.auto_arima(train_df[[label]],
                            X=train_df.loc[:, train_df.columns != label],
                            m=12,
                            suppress_warnings=True,
                            trace=True)
    sxmodel.summary()

    result = test_df[[label]][:]
    result[label + ' (ARIMAX)'] = sxmodel.predict(output_steps, test_df.loc[:, test_df.columns != label])
    result.plot()
    plt.title(f"mean absolute error: {mean_absolute_error(result[label], result[label + ' (ARIMAX)'])}")
    plt.savefig('images/prediction_ARIMAX_' + label + '.eps')
    plt.show()

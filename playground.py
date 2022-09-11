from util import *
from LSTM import *
from sklearn.metrics import mean_absolute_error
import pmdarima as pm

if __name__ == "__main__":
    label = 'y'

    df = get_test_curve(mean=150, amplitude=100)
    df.plot()
    plt.savefig('images/' + label + '.eps')
    plt.show()

    fft_analysis(df, samples_per_day=1 / 30.437)

    output_steps = 12

    train_df = df[:-output_steps]
    test_df = df[-output_steps:]

    train_mean = train_df.mean()
    train_std = train_df.std()

    train_df = (train_df - train_mean) / train_std
    test_df = (test_df - train_mean) / train_std

    # phase = train_df.index.map(pd.Timestamp.timestamp) * np.pi / 365.2425 / 12 / 60 / 60
    phase = train_df.index.map(pd.Timestamp.timestamp) * np.pi / 365.2425 / 12 / 60 / 60
    train_df['sin'] = np.sin(phase)
    train_df['cos'] = np.cos(phase)
    train_df.plot()
    plt.show()

    phase = test_df.index.map(pd.Timestamp.timestamp) * np.pi / 365.2425 / 12 / 60 / 60
    test_df['sin'] = np.sin(phase)
    test_df['cos'] = np.cos(phase)
    test_df.plot()
    plt.show()

    sxmodel = pm.auto_arima(train_df[[label]],
                            X=train_df.loc[:, train_df.columns != label],
                            m=12,
                            suppress_warnings=True,
                            trace=True)
    sxmodel.summary()

    prediction = sxmodel.predict(12, test_df.loc[:, test_df.columns != label])
    test_df[label + ' (SARIMAX)'] = prediction
    test_df.plot()
    plt.savefig('sx_predict_' + label + '.eps')
    plt.show()

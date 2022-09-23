import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt


def get_test_curve(length=120, mean=0, amplitude=10, period=12, freq='M', noise=0.05):
    x = np.arange(length)
    y = mean + amplitude * np.sin(2 * np.pi / period * x)
    y = np.random.normal(y, noise * y)
    df = pd.DataFrame(data={'y': y},
                      index=pd.date_range(end=pd.Timestamp.today(), periods=length, freq=freq))
    return df


def fft_analysis(df, label='y', samples_per_day=1.):
    fft = tf.signal.rfft(df[label])
    f_per_dataset = np.arange(0, len(fft))
    n_samples = len(df[label])
    years_per_dataset = n_samples / (samples_per_day * 365.2524)
    f_per_year = f_per_dataset / years_per_dataset
    plt.step(f_per_year, np.abs(fft))
    plt.xscale('log')
    plt.ylim(0, 1000)
    plt.xlim([0.1, max(plt.xlim())])
    plt.xticks([1, 30.437, 365.2524], labels=['1/year', '1/month', '1/day'])
    plt.xlabel('Frequency (log scale)')
    plt.savefig('images/fft_' + label + '.eps')
    plt.show()


def split(df, input_steps, output_steps, val_rate=0.1):
    val_size = int(val_rate * len(df))
    return df[:-output_steps - val_size + input_steps], \
           df[-output_steps - val_size: -output_steps], \
           df[-input_steps - output_steps:]


def normalize(train_df, val_df, test_df):
    train_mean = train_df.mean()
    train_std = train_df.std()
    return (train_df - train_mean) / train_std,\
           (val_df - train_mean) / train_std,\
           (test_df - train_mean) / train_std


def add_trigonometric_input(df, period='Y'):
    phase = df.index.map(pd.Timestamp.timestamp) * np.pi / 12 / 60 / 60
    if period == 'D':
        df['day sin'] = np.sin(phase)
        df['day cos'] = np.cos(phase)
    phase /= 365.2425
    df['year sin'] = np.sin(phase)
    df['year cos'] = np.cos(phase)
    return df

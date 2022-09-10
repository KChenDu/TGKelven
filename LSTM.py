import numpy as np
import tensorflow as tf


def make_dataset(data, input_steps, output_steps, label, batch_size=64):
    inputs = data[:-output_steps]
    targets = np.lib.stride_tricks.sliding_window_view(data[[label]][input_steps:],
                                                       output_steps,
                                                       axis=0).transpose((0, 2, 1))
    return tf.keras.utils.timeseries_dataset_from_array(inputs,
                                                        np.array(targets),
                                                        input_steps,
                                                        batch_size=batch_size,
                                                        shuffle=True)


def lstm(train_df, val_df, input_steps, output_steps, label='y', lstm_units=32, epochs=20):
    multi_lstm_model = tf.keras.Sequential([
        # Shape [batch, time, features] => [batch, lstm_units].
        # Adding more `lstm_units` just overfits more quickly.
        tf.keras.layers.LSTM(lstm_units),
        # Shape => [batch, out_steps * features].
        tf.keras.layers.Dense(output_steps)])
    multi_lstm_model.compile(loss=tf.keras.losses.MeanSquaredError(),
                             optimizer=tf.keras.optimizers.Adam(),
                             metrics=[tf.keras.metrics.MeanAbsoluteError()])
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                      patience=3,
                                                      mode='min')
    history = multi_lstm_model.fit(make_dataset(train_df, input_steps, output_steps, label),
                                   epochs=epochs,
                                   validation_data=make_dataset(val_df, input_steps, output_steps, label),
                                   callbacks=[early_stopping])
    return multi_lstm_model, history

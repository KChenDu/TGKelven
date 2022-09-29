import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from util import save_figure


def make_dataset(data, input_steps, output_steps, label, batch_size=32):
    inputs = data[:-output_steps]
    targets = np.lib.stride_tricks.sliding_window_view(data[[label]][input_steps:],
                                                       output_steps,
                                                       axis=0).transpose((0, 2, 1))
    return tf.keras.utils.timeseries_dataset_from_array(inputs,
                                                        targets,
                                                        input_steps,
                                                        batch_size=batch_size,
                                                        shuffle=True)


class LSTM:
    def __init__(self, train_df, val_df, input_steps, output_steps, label='y', lstm_units=32, epochs=20, batch_size=32):
        model = tf.keras.Sequential([
            # Shape [batch, time, features] => [batch, lstm_units].
            # Adding more `lstm_units` just overfits more quickly.
            tf.keras.layers.LSTM(lstm_units),
            # Shape => [batch, out_steps * features].
            tf.keras.layers.Dense(output_steps)])
        model.compile(loss=tf.keras.losses.MeanSquaredError(),
                      optimizer=tf.keras.optimizers.Adam(),
                      metrics=[tf.keras.metrics.MeanAbsoluteError()])
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                          patience=3,
                                                          mode='min')
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath='models/checkpoints_LSTM_' + label,
                                                                       save_best_only=True,
                                                                       mode='min')
        history = model.fit(make_dataset(train_df, input_steps, output_steps, label, batch_size),
                            epochs=epochs,
                            validation_data=make_dataset(val_df, input_steps, output_steps, label, batch_size),
                            callbacks=[early_stopping, model_checkpoint_callback])
        model.save('models/model_LSTM_' + label)
        self.model = model
        self.history = history
        self.label = label
        self.input_steps = input_steps
        self.output_steps = output_steps

    def show_history(self):
        plt.plot(self.history.history['loss'])
        plt.plot(self.history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'])
        save_figure(self.label + '_LSTM_history')

    def predict(self, test_df):
        return self.model.predict(make_dataset(test_df, self.input_steps, self.output_steps, self.label))[0]


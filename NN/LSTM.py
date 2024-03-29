import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from util import save_figure


def make_dataset(data, input_steps, label, batch_size=32):
    inputs = data[:-1]
    targets = np.lib.stride_tricks.sliding_window_view(data[[label]][input_steps:],
                                                       1,
                                                       axis=0).transpose((0, 2, 1))
    return tf.keras.utils.timeseries_dataset_from_array(inputs,
                                                        targets,
                                                        input_steps,
                                                        batch_size=batch_size)


class LSTM:
    def __init__(self, train_df, val_df, input_steps, output_steps, label='y', lstm_units=32, epochs=20, batch_size=32, patience=5):
        model = tf.keras.Sequential([
            # Shape [batch, time, features] => [batch, lstm_units].
            # Adding more `lstm_units` just overfits more quickly.
            tf.keras.layers.LSTM(lstm_units),
            # Shape => [batch, out_steps * features].
            tf.keras.layers.Dense(1)])
        model.compile(loss=tf.keras.losses.MeanSquaredError(),
                      optimizer=tf.keras.optimizers.Adam(),
                      metrics=[tf.keras.metrics.MeanAbsoluteError()])
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                          patience=patience,
                                                          mode='min')
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath='models/checkpoints_LSTM_' + label,
                                                                       save_best_only=True,
                                                                       mode='min')
        history = model.fit(make_dataset(train_df, input_steps, label, batch_size),
                            epochs=epochs,
                            validation_data=make_dataset(val_df, input_steps, label, batch_size),
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
        result = []
        label = self.label
        input_steps = self.input_steps
        model = self.model
        output_steps = self.output_steps
        inputs = test_df[:input_steps]
        for i in range(output_steps):
            prediction = model.predict(tf.keras.utils.timeseries_dataset_from_array(inputs, None, input_steps, batch_size=1))[0]
            inputs = inputs[1:]
            index = test_df.index[input_steps + i]
            inputs.loc[index] = test_df.loc[index, :]
            inputs.at[index, label] = prediction
            result.append(prediction)
        return np.array(result).transpose()[0]

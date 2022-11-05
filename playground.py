from util import get_test_curve, Normalizer, lstm_split, add_trigonometric_input
from NN.LSTM import LSTM
from NN.LSTM import make_dataset

label = 'y'
input_steps = 12
output_steps = 12

df = get_test_curve(360, 150, 100)

normalizer = Normalizer(df[:-output_steps])

train_df, val_df, test_df = lstm_split(df, input_steps, output_steps)

train_df = add_trigonometric_input(normalizer.normalize(train_df))
val_df = add_trigonometric_input(normalizer.normalize(val_df))
test_df = add_trigonometric_input(normalizer.normalize(test_df))

lstm = LSTM(train_df, val_df, input_steps, output_steps, lstm_units=8, epochs=3)
lstm.show_history()

output = lstm.predict(test_df)

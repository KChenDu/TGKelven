from util import *
from NARX import NARMAX

label = 'y'

df = get_test_curve(480, 150, 100)
df.plot()
save_figure(label)

fft_analysis(df, samples_per_day=1 / 30.437)

output_steps = 12
train_df, val_df, test_df = simple_split(df, output_steps)
train_df, val_df, test_df = normalize(train_df, test_df, val_df)

train_df = add_trigonometric_input(train_df)
train_df.plot()
save_figure('train_NARX_' + label)

val_df = add_trigonometric_input(val_df)
val_df.plot()
save_figure('val_NARX_' + label)

test_df = add_trigonometric_input(test_df)
test_df.plot()
save_figure('test_NARX_' + label)

narmax = NARMAX(train_df, val_df)

result = test_df[[label]]
result[label + ' (NARX)'] = narmax.predict(test_df)
result.plot()
save_figure('prediction_NARX_' + label)

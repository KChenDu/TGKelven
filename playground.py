from util import *
from NARX import NARMAX

label = 'y'

df = get_test_curve(480, 150, 100)
df.plot()
fig_type = 'eps'
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

train_df = add_trigonometric_input(train_df)
train_df.plot()
plt.savefig('images/train_NARX_' + label + '.eps')
plt.show()

test_df = add_trigonometric_input(test_df)
test_df.plot()
plt.savefig('images/test_NARX_' + label + '.eps')
plt.show()

narmax = NARMAX(train_df, test_df)

result = narmax.predict(test_df)

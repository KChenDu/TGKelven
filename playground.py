import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from util import normalize

df = pd.read_csv('jena_climate_2009_2016.csv')
# Slice [start:stop:step], starting from index 5 take every 6th record.
# one measure 6 days / 5 measures a month
df = df[5::6 * 24 * 7]
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

# Caution: Change this line when changes frequency
df = df.resample('1M').mean().interpolate()

# Selection of columns
df = df[['T (degC)',
         #'p (mbar)',
         #'rh (%)',
         #'VPmax (mbar)',
         #'VPact (mbar)',
         #'VPdef (mbar)',
         #'sh (g/kg)',
         #'H2OC (mmol/mol)',
         #'rho (g/m**3)',
         #'Wx',
         #'Wy',
         #'max Wx',
         #'max Wy',
         ]]

df = normalize(df)

df[-100:].plot()
plt.show()

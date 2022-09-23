from torch import nn
from sysidentpy.basis_function import Polynomial
from sysidentpy.neural_network import NARXNN
from sysidentpy.utils.narmax_tools import regressor_code
import pandas as pd
import numpy as np


class NARX(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.layer1 = nn.Linear(n_features, 30)
        self.layer2 = nn.Linear(30, 30)
        self.layer3 = nn.Linear(30, 1)
        self.tanh = nn.Tanh()

    def forward(self, input):
        output = self.layer1(input)
        output = self.tanh(output)
        output = self.layer2(output)
        output = self.tanh(output)
        return self.layer3(output)


class NARMAX:
    def __init__(self, train_df, val_df, label='y', xlag=2, ylag=2, polynomial_degree=2):
        basis_function = Polynomial(polynomial_degree)
        lag = range(1, xlag + 1)
        xlags = []
        n = train_df.shape[1] - 1
        for i in range(n):
            xlags.append(lag)
        model = NARXNN(xlag=xlags,
                       ylag=ylag,
                       basis_function=basis_function,
                       batch_size=32,
                       epochs=20,
                       net=NARX(regressor_code(X=train_df.loc[:, train_df.columns != label],
                                               xlag=xlags,
                                               ylag=2,
                                               model_representation="neural_network",
                                               basis_function=basis_function).shape[0]),
                       optim_params={'betas': (0.9, 0.999),
                                     'eps': 1e-05})
        self.model = model.fit(X=train_df.loc[:, train_df.columns != label].to_numpy(),
                               y=train_df[[label]].to_numpy(),
                               X_test=val_df.loc[:, val_df.columns != label].to_numpy(),
                               y_test=val_df[[label]].to_numpy())
        self.label = label

    def predict(self, test_df):
        label = self.label
        return pd.DataFrame({label + ' (NARX)': np.ravel(np.array(self.model.predict(X=test_df.loc[:, test_df.columns != label].to_numpy(),
                                                                                     y=test_df[[label]].to_numpy())))}, test_df.index)

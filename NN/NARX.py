from torch import nn
from sysidentpy.basis_function import Polynomial
from sysidentpy.neural_network import NARXNN
from sysidentpy.utils.narmax_tools import regressor_code
import torch

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
    def __init__(self, train_df, val_df=None, label='y', xlag=2, ylag=2, polynomial_degree=2):
        basis_function = Polynomial(polynomial_degree)
        lag = list(range(1, xlag + 1))
        n = train_df.shape[1] - 1
        if (n > 1):
            xlags = []
            for i in range(n):
                xlags.append(lag)
        else:
            xlags = xlag
        model = NARXNN(xlag=xlags,
                       ylag=ylag,
                       basis_function=basis_function,
                       batch_size=32,
                       epochs=20,
                       net=NARX(regressor_code(X=train_df.loc[:, train_df.columns != label],
                                               xlag=xlags,
                                               ylag=ylag,
                                               model_representation="neural_network",
                                               basis_function=basis_function).shape[0]),
                       optim_params={'betas': (0.9, 0.999),
                                     'eps': 1e-05})
        model.fit(X=train_df.loc[:, train_df.columns != label].to_numpy(),
                  y=train_df[[label]].to_numpy(),
                  X_test=val_df.loc[:, val_df.columns != label].to_numpy(),
                  y_test=val_df[[label]].to_numpy())
        torch.save(model.net.state_dict(), 'models/model_NARX_' + label + '.pt')
        self.label = label
        self.model = model

    def predict(self, test_df):
        label = self.label
        return self.model.predict(X=test_df.loc[:, test_df.columns != label].to_numpy(),
                                  y=test_df[[label]].to_numpy()).flatten()

    def load(self):
        self.model.net.load_state_dict(torch.load('models/model_NARX_' + self.label + '.pt'))

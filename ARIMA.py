import pmdarima as pm
import pickle


class ARIMA:
    def __init__(self, train_df, label='y', period=12):
        model = pm.auto_arima(train_df[[label]],
                              X=train_df.loc[:, train_df.columns != label],
                              m=period,
                              suppress_warnings=True,
                              trace=True)
        model.summary()
        with open('models/model_ARIMA_' + label + '.pkl', 'wb') as pkl:
            pickle.dump(model, pkl)
        self.model = model
        self.label = label

    def predict(self, test_df, output_steps):
        return self.model.predict(output_steps, test_df.loc[:, test_df.columns != self.label])

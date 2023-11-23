import pandas as pd
from sklearn.model_selection import train_test_split
import h2o
from h2o.automl import H2OAutoML


class Model_Manager:
    def __init__(self, file=None):
        self._data_set = (
            self._read_data_set_from_file(file) if file is not None else None
        )
        if file is not None:
            (
                self._feature_train,
                self._feature_test,
                self._label_train,
                self._label_test,
            ) = self._split_data_into_training_and_testing_sets()

    def _read_data_set_from_file(self, file):
        return pd.read_csv(file, sep="\t", header=None, names=["Label", "Message"])

    def get_data_set(self):
        return self._data_set

    def get_data_set_head(self):
        return self._data_set.head()

    def _split_data_into_training_and_testing_sets(self):
        return train_test_split(
            self._data_set["Message"],
            self._data_set["Label"],
            test_size=0.2,
            random_state=42,
        )

    def get_train_data_set(self):
        return self._feature_train, self._label_train

    def get_test_data_set(self):
        return self._feature_test, self._label_test

    def use_saved_model(self, path):
        h2o.init()
        return h2o.load_model(path)

    def make_model_with_training_data(
        self, data_feature=None, data_label=None, max_models=20, max_runtime_secs=3600
    ):
        data_feature = self._feature_train if data_feature is None else data_feature
        data_label = self._label_train if data_label is None else data_label
        
        h2o.init()
        
        h2o_train = h2o.H2OFrame(pd.concat([data_feature, data_label], axis=1))
        
        aml = H2OAutoML(
            max_models=max_models,
            seed=1,
            balance_classes=True,
            max_runtime_secs=max_runtime_secs,
            include_algos=["GBM", "XGBoost", "DeepLearning"],
        )

        feature = ["Message"]
        target = "Label"

        aml = H2OAutoML(max_models=10, seed=1)
        aml.train(x=feature, y=target, training_frame=h2o_train)

        return aml.leader

    def make_predictions(self, model, data_feature=None, data_label=None):
        data_feature = self._feature_test if data_feature is None else data_feature
        data_label = self._label_test if data_label is None else data_label
        h2o_test = h2o.H2OFrame(pd.concat([data_feature, data_label], axis=1))

        predictions = model.predict(h2o_test)

        return predictions

    def make_single_prediction(self, model, single_message="hello this is a message"):
        df_single = pd.DataFrame([single_message], columns=["Message"])
        h2o_single_message = h2o.H2OFrame(df_single)

        prediction = model.predict(h2o_single_message)

        return prediction.as_data_frame().iloc[0, 0]

    def evaluate_performance(self, model, data_feature=None, data_label=None):
        data_feature = self._feature_test if data_feature is None else data_feature
        data_label = self._label_test if data_label is None else data_label
        h2o_test = h2o.H2OFrame(pd.concat([data_feature, data_label], axis=1))

        performance = model.model_performance(h2o_test)
        return performance

    def save_model(self, model, path):
        model_path = h2o.save_model(model=model, path=path, force=True)
        print("Model saved to: " + model_path)
        return model_path

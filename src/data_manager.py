import pandas as pd
from sklearn.model_selection import train_test_split


class Data_Manager:
    def __init__(self, file):
        self._data_set = self._read_data_set_from_file(file)
        (
            self._feature_train,
            self._feature_test,
            self._label_train,
            self._label_test,
        ) = self._split_data_into_training_and_testing_sets()

    def _read_data_set_from_file(self, file):
        # Replace 'path_to_your_data_file' with the actual file path
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

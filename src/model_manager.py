import pandas as pd
import re
import h2o
from textblob import TextBlob
from sklearn.model_selection import train_test_split
from h2o.automl import H2OAutoML


class Model_Manager:
    def __init__(self, file=None):
        self._data_set = (
            self._process_data_set_from_file(file) if file is not None else None
        )
        if file is not None:
            (
                self._feature_train,
                self._feature_test,
                self._label_train,
                self._label_test,
            ) = self._split_data_into_training_and_testing_sets()
        
        self._chosen_features = [
            "Message",
            "Message_Length",
            "Uppercase_Count",
            "Count_Punctuations",
            "URL_Count",
            "Digit_Count",
        ]

    def _process_data_set_from_file(self, file):
        df = pd.read_csv(file, sep="\t", header=None, names=["Label", "Message"])

        df = self._create_more_features(df=df)

        return df

    def _create_more_features(self, df):
        df["Message_Length"] = df["Message"].apply(self._message_length)
        df["Word_Count"] = df["Message"].apply(self._count_of_words)
        df["Character_Count"] = df["Message"].apply(self._count_of_characters)
        df["Uppercase_Count"] = df["Message"].apply(self._count_uppercase_words)
        df["Count_Punctuations"] = df["Message"].apply(self._count_punctuation)
        df["URL_Count"] = df["Message"].apply(self._count_urls)
        df["Digit_Count"] = df["Message"].apply(self._count_digits)
        df["Special_Char_Count"] = df["Message"].apply(self._count_special_characters)
        df["Sentiment_Score"] = df["Message"].apply(self._sentiment_score)
        
        return df
    
    def _message_length(self, message):
        return len(message)

    def _count_of_words(self, message):
        return len(message.split())

    def _count_of_characters(self, message):
        return sum(len(word) for word in message.split())

    def _count_punctuation(self, message, punctuation="!?"):
        return sum(message.count(char) for char in punctuation)

    def _count_uppercase_words(self, message):
        return sum(word.isupper() for word in message.split())

    def _count_urls(self, message):
        url_pattern = r"https?://\S+|www\.\S+"
        return len(re.findall(url_pattern, message))

    def _count_digits(self, message):
        return sum(char.isdigit() for char in message)

    def _count_special_characters(self, message, characters="@#$%&"):
        return sum(message.count(char) for char in characters)

    def _sentiment_score(self, message):
        return TextBlob(message).sentiment.polarity

    def print_pearson_correlation(self):
        df = self._data_set

        label_map = {"spam": 1, "ham": 0}
        df["Label"] = df["Label"].map(label_map)

        correlations = df.corr()
        target_correlations = correlations["Label"].sort_values(ascending=False)

        print(target_correlations)
        
        features = df[['Character_Count', 'Message_Length', 'Word_Count']]
        correlation_matrix = features.corr()

        print('multicollinearity:\n')
        print(correlation_matrix)

    def get_data_set(self):
        return self._data_set

    def get_data_set_head(self):
        return self._data_set.head()

    def _split_data_into_training_and_testing_sets(self):
        feature_columns = [
            "Message",
            "Message_Length",
            "Word_Count",
            "Character_Count",
            "Uppercase_Count",
            "Count_Punctuations",
            "URL_Count",
            "Digit_Count",
            "Sentiment_Score",
        ]
        return train_test_split(
            self._data_set[feature_columns],
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

        target = "Label"

        aml = H2OAutoML(max_models=10, seed=1)
        aml.train(x=self._chosen_features, y=target, training_frame=h2o_train)

        return aml.leader

    def make_predictions(self, model, data_feature=None, data_label=None):
        data_feature = self._feature_test if data_feature is None else data_feature
        data_label = self._label_test if data_label is None else data_label
        h2o_test = h2o.H2OFrame(pd.concat([data_feature, data_label], axis=1))

        predictions = model.predict(h2o_test)

        return predictions

    def make_single_prediction(self, model, single_message="hello this is a message"):
        df_single = pd.DataFrame([single_message], columns=["Message"])
        df_single = self._create_more_features(df=df_single)
        
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

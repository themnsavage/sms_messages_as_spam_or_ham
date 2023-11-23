import pandas as pd
import re
import nltk
import h2o
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from h2o.automl import H2OAutoML
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download necessary NLTK datasets
nltk.download('punkt')
nltk.download('stopwords')

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
        df = pd.read_csv(file, sep="\t", header=None, names=["Label", "Message"])
        df["Tokens"] = df["Message"].apply(self._filter_message)
        return df

    def get_data_set(self):
        return self._data_set

    def get_data_set_head(self):
        return self._data_set.head()

    def _split_data_into_training_and_testing_sets(self):
        return train_test_split(
            self._data_set["Tokens"],
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
        # Tokenize and transform the feature data
        tfidf_vectorizer = TfidfVectorizer()
        tfidf_features = tfidf_vectorizer.fit_transform(data_feature)

        joblib.dump(tfidf_vectorizer, 'fitted_tfidf_vectorizer.pkl')    
        # Convert to DataFrame
        tfidf_df = pd.DataFrame(tfidf_features.toarray(), columns=tfidf_vectorizer.get_feature_names_out())

        # Concatenate with the label
        data_for_h2o = pd.concat([tfidf_df, data_label.reset_index(drop=True)], axis=1)

        h2o.init()

        # Create H2OFrame
        h2o_train = h2o.H2OFrame(data_for_h2o)

        feature = tfidf_vectorizer.get_feature_names_out().tolist()
        target = "Label"

        aml = H2OAutoML(max_models=max_models, seed=1, balance_classes=True, max_runtime_secs=max_runtime_secs)
        aml.train(x=feature, y=target, training_frame=h2o_train)

        return aml.leader

    def make_predictions(self, model, data_feature=None, data_label=None):
        data_feature = self._feature_test if data_feature is None else data_feature
        data_label = self._label_test if data_label is None else data_label
        h2o_test = h2o.H2OFrame(pd.concat([data_feature, data_label], axis=1))

        predictions = model.predict(h2o_test)

        return predictions

    def make_single_prediction(self, model, single_message="hello this is a message"):
        tfidf_vectorizer = joblib.load('fitted_tfidf_vectorizer.pkl')
        # Preprocess the message
        # (You should implement the _filter_message method similar to how you preprocessed your training data)
        processed_message = self._filter_message(single_message)

        # Transform the message using the loaded TF-IDF vectorizer
        tfidf_features = tfidf_vectorizer.transform([processed_message])

        # Convert to H2OFrame
        h2o_message = h2o.H2OFrame(pd.DataFrame(tfidf_features.toarray()))

        # Make prediction
        prediction = model.predict(h2o_message)

        # Convert the prediction to a readable format
        predicted_label = prediction.as_data_frame().iloc[0, 0]

        return predicted_label

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
    


    def _filter_message(self, message):
        # Convert to lowercase
        message = message.lower()

        # Remove punctuation and non-word characters
        message = re.sub(r'[^\w\s]', '', message)

        # Tokenize the message
        tokens = word_tokenize(message)

        # Optionally: Remove stop words
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words]

        # Optionally: Apply stemming
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(word) for word in tokens]

        return ' '.join(tokens)


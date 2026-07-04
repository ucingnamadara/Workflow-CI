import pandas as pd
import numpy as np
import mlflow

from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from gensim.models import Word2Vec
from sklearn.utils.class_weight import compute_class_weight
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

def build_svc_model():
    print("Start building SVC model...")
    df = load_data()
    X, y_int, y_one_hot, encoder = prepare_feature_and_label(df)
    X_train, X_test, y_train, y_test = split_data(X, y_int)
    training_svc_model(X_train, y_train, X_test, y_test)


def load_data():
    print("Loading data...")
    df = pd.read_csv('dataset_preprocessing.csv')
    return df

def prepare_feature_and_label(df):
    print("Preparing features and labels...")
    X = df['final'].apply(lambda x: " ".join(x) if isinstance(x, list) else str(x))
    y = df['Genre']

    # Encode label
    encoder = LabelEncoder()
    y_int = encoder.fit_transform(y)
    y_one_hot = to_categorical(y_int)

    print("Class map:", {cls: idx for idx, cls in enumerate(encoder.classes_)})
    return X, y_int, y_one_hot, encoder

def split_data(X, y_int,test_size=0.20, random_state=42):
    print("Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_int, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test

def training_svc_model(X_train, y_train, X_test, y_test):
    print("Training SVC model...")

    initialize_mlflow()

    X_train_tfidf, X_test_tfidf = init_tfidf_vectorizer(X_train, X_test)

    print("Starting MLflow run for SVC model...")
    with mlflow.start_run(run_name="Tfidf_Svc_Model"):
        print("Training SVC model with TF-IDF features...")
        svm_model = SVC(kernel='linear', class_weight='balanced', random_state=42)
        svm_model.fit(X_train_tfidf, y_train)
        
        print("Evaluating model on test data...")
        y_pred = svm_model.predict(X_test_tfidf)
        accuracy_test = accuracy_score(y_test, y_pred)
        
        report_test = classification_report(y_test, y_pred)
        print(f"Accuracy Test : {accuracy_test}")
        print(f"Classification Report : {report_test}")

        mlflow.sklearn.log_model(
            sk_model=svm_model,
            artifact_path="model",
            serialization_format="cloudpickle"
        )
    

def init_tfidf_vectorizer(X_train, X_test):
    print("Initializing TF-IDF vectorizer...")
    tfidf = TfidfVectorizer()
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)
    return X_train_tfidf, X_test_tfidf

def initialize_mlflow():
    print("Initializing MLflow...")

if __name__ == "__main__":
    mlflow.sklearn.autolog()
    build_svc_model()
import src.io_utils as io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA


from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    roc_auc_score
)

class TrainingTools:
    def __init__(self):
        
        df = io.load_data("data/housing.csv")
        self.df = df



    def prepare_data(self, df, top_n=20):
        """
        Prepare data for training. Adds new features and creates a binary target variable based on the top_n percentage of median house values.
        """
    
        df["rooms_per_household"] = df["total_rooms"] / df["households"]
        df["bedrooms_per_household"] = df["total_bedrooms"] / df["households"]

        threshold = df["median_house_value"].quantile(1-(top_n / 100))
        df["target"] = (df["median_house_value"] >= threshold).astype(int)

        return df
    
    def preprocess_data(self, numeric_features, categorical_features):
        """
        Preprocesses the data by creating pipelines for numeric and categorical features, including imputation and scaling/encoding, and combines them into a ColumnTransformer.
        """

        numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        (   "scaler", StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False))
        ])

        preprocess = ColumnTransformer(
            transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
                ],
            remainder="drop"
            )
        
        return preprocess
    

    def evaluate_on_test(self, model, X_test, y_test, target_names):
        """
        Evaluates the model on the test set and prints out various performance metrics including accuracy, precision, recall, F1 score, confusion matrix, and classification report.
        """

        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)


        print(f"\n=========================================")
        print(f"{model.__class__.__name__} - TEST")
        print(f"\n=========================================")
        print(f"Accuracy  : {acc:.3f}")
        print(f"Precicion : {prec:.3f}")
        print(f"Recall    : {rec:.3f}")
        print(f"F1        : {f1:.3f}")


        cm = confusion_matrix(y_test, y_pred)

        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)

        disp.plot(cmap="RdPu")
        plt.title(f"Confusion matrix - {model.named_steps['model'].__class__.__name__} (TEST)")
        plt.show()

        print("\nDetaljer per klass (classification_report)")
        print(classification_report(y_test, y_pred, target_names=target_names, zero_division=0))

        tn, fp, fn, tp = cm.ravel()
        print(f"TN={tn}, FP={fp}, FN={fn}, TP={tp}")
        print("Tolkning")
        print("- FP: vi flaggade positivt med det var negativt")
        print("- FN: vi missade positivt och flaggade det som negativt")

        return { "accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "cm": cm }
    

    
pca = PCA(n_components=10)  # Keep all components for explained variance analysis

from sklearn.base import BaseEstimator, TransformerMixin

class ColumnDropper(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.drop(columns=self.columns, errors="ignore")
    

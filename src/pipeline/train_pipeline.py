# Example training pipeline (optional)
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle
import os


class TrainingPipeline:

    def train(self, df):
        X = df.drop("math_score", axis=1)
        y = df["math_score"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = LinearRegression()
        model.fit(X_train, y_train)

        # Save model
        os.makedirs("artifacts", exist_ok=True)
        pickle.dump(model, open("artifacts/model.pkl", "wb"))

        print("Model training complete!")



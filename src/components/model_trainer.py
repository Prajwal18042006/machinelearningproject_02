# import os 
# import sys
# from dataclasses import dataclass
# from catboost import CatBoostRegressor 
# from sklearn.ensemble import (
#     AdaBoostRegressor,
#     GradientBoostingRegressor,
#     RandomForestRegressor


# )

# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import r2_score
# from sklearn.neighbors import KNeighborsRegressor
# from sklearn.tree import DecisionTreeRegressor
# from xgboost import XGBRegressor 
# from src.exception import CustomException 
# from src.logger import logging

# from src.utils import save_obj,evaluate_model
# @dataclass
# class modeltrainerconfig:
#     trained_model_file_path=os.path.join("artifacts","model.pkl")
# class modeltrainer:
#     def __init__(self):
#         self.model_trainer_config=modeltrainerconfig()

#     def initiatemodeltrainer(self,train_arr,test_arr):

#         try:
#             logging.info("spliting raining test inout data :")
#             x_train,y_train,x_test,y_test=(
#                 train_arr[:,:-1],
#                 train_arr[:,-1],
#                 test_arr[:,:-1],
#                 test_arr[:,-1],
#             )
#             models={
#                 "random forest":RandomForestRegressor(),
#                 "decision tree":DecisionTreeRegressor(),
#                 "liner regression":LinearRegression(),
#                 "gradient boosting":GradientBoostingRegressor(),
#                 "k neighbors classifire":KNeighborsRegressor(),
#                 "XGBclassifier":XGBRegressor(),
#                 "catboosting classifier":CatBoostRegressor(verbose=False),
#                 "adaboost classfier":AdaBoostRegressor()


#             }
#             model_report=evaluate_model(x_train=x_train,y_train=y_train,y_test=y_test,x_test=x_test,models=models)

#             ##to get a best model score from dict 
#             best_model_score=max(sorted(model_report.values()))
#             #to get a best model name 
#             best_model_name=list(model_report.keys())[
#                 list(model_report.values().index(best_model_score))
#             ]
#             best_model=models[best_model_name]
#             if best_model_score<0.6:
#                 raise CustomException("no best model found ")
#             logging.info("best found model on both tarining and testing datasdet ")
#             save_obj(
#                 file_path=self.model_trainer_config.trained_model_file_path,
#                 obj=best_model
#             )
#             predicted=best_model.predict(x_test)
#             r2_square=r2_score(y_test,predicted)
#             return r2_score

#         except:
#             pass 
import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor 
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor 

from src.exception import CustomException 
from src.logger import logging
from src.utils import save_obj, evaluate_model


@dataclass
class modeltrainerconfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")


class modeltrainer:
    def __init__(self):
        self.model_trainer_config = modeltrainerconfig()

    def initiatemodeltrainer(self, train_arr, test_arr):

        try:
            logging.info("Splitting training and test input data")

            x_train, y_train, x_test, y_test = (
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1],
            )

            models = {
                "RandomForest": RandomForestRegressor(),
                "DecisionTree": DecisionTreeRegressor(),
                "LinearRegression": LinearRegression(),
                "GradientBoosting": GradientBoostingRegressor(),
                "KNN": KNeighborsRegressor(),
                "XGBoost": XGBRegressor(),
                "CatBoost": CatBoostRegressor(verbose=False),
                "AdaBoost": AdaBoostRegressor()
            }

            params = {

                "Decision Tree": {
                    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    'splitter': ['best', 'random'],
                },

                "Random Forest": {
                    'n_estimators': [8, 16, 32, 64, 128, 256],
                },

                "Gradient Boosting": {
                    'learning_rate': [1.0, 0.1, 0.05, 0.01],
                    'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },

                "Linear Regression": {},

                "K-Neighbour Regressor": {
                    'n_neighbors': [5, 7, 9, 11],
                    # 'weights': ['uniform', 'distance'],
                    # 'algorithm': ['ball_tree', 'kd_tree', 'brute'],
                },

                "XGBRegressor": {
                    'learning_rate': [1.0, 0.1, 0.05, 0.01],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },

                "CatBoost Regressor": {
                    'depth': [6, 8, 10],
                    'learning_rate': [0.01, 0.1, 0.05],
                    'iterations': [30, 50, 100]
                },

                "AdaBoost Regressor": {
                    'learning_rate': [1.0, 0.1, 0.5, 0.01],
                    'loss': ['linear', 'square', 'exponential'],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                }
            }

            # Evaluate all models
            model_report = evaluate_model(
                x_train=x_train,
                y_train=y_train,
                x_test=x_test,
                y_test=y_test,
                models=models,
                params=params
            )

            # Best model score
            best_model_score = max(model_report.values())
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            # Print best model details
            print("\n============================")
            print(f" BEST MODEL NAME  : {best_model_name}")
            print(f" BEST MODEL SCORE : {best_model_score}")
            print("============================\n")

            if best_model_score < 0.6:
                raise CustomException("No good model found!")

            # Save best model
            save_obj(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            # Predictions
            predicted = best_model.predict(x_test)
            r2_square = r2_score(y_test, predicted)

            return r2_square

        except Exception as e:
            raise CustomException(e, sys)

# import os  
# import dill 

# import sys
# import pandas as pd 
# import numpy as np 
# from src.exception import CustomException
# from sklearn.metrics import r2_score
# from sklearn.model_selection import GridSearchCV
# def save_obj(file_path,obj):
#     try:
#         dir_path=os.path.dirname(file_path)
#         os.makedirs(dir_path,exist_ok=True)

#         with open(file_path,"wb")as file_obj:
#             dill.dump(obj,file_obj)
#     except Exception as e:
#         raise CustomException(e,sys)

# def evaluate_model(x_train, y_train, x_test, y_test, models,params):
#     try:
#         report = {}

#         for model_name, model in models.items():
#             #model.fit(x_train, y_train)

#             # Predictions
#             y_train_pred = model.predict(x_train)
#             y_test_pred = model.predict(x_test)

#             # Scores
#             train_score = r2_score(y_train, y_train_pred)
#             test_score = r2_score(y_test, y_test_pred)

#             report[model_name] = test_score

#         return report

#     except Exception as e:
#         raise CustomException(e, sys)


import os  
import dill 
import sys
import pandas as pd 
import numpy as np 
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException


def save_obj(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)



def evaluate_model(x_train, y_train, x_test, y_test, models, params):
    try:
        report = {}

        model_names = list(models.keys())
        model_list = list(models.values())

        for i in range(len(model_list)):

            model = model_list[i]
            model_name = model_names[i]

            print(f"\n🔍 Running GridSearchCV for: {model_name}")

            # Get parameter grid for this model
            param_grid = params.get(model_name, {})

            # ---- GRID SEARCH START ----
            if len(param_grid) != 0:   # only run tuning if parameters exist

                gs = GridSearchCV(model, param_grid, cv=3)
                gs.fit(x_train, y_train)

                # Set best parameters
                model.set_params(**gs.best_params_)

            # Fit the model with final parameters
            model.fit(x_train, y_train)
            # ---- GRID SEARCH END ----

            # Predictions
            y_train_pred = model.predict(x_train)
            y_test_pred = model.predict(x_test)

            # Model performance
            train_score = r2_score(y_train, y_train_pred)
            test_score = r2_score(y_test, y_test_pred)

            print(f"📌 Best Test R² for {model_name}: {test_score}")

            report[model_name] = test_score

        return report

    except Exception as e:
        raise CustomException(e, sys)

def load_obj(file_path):
    try:
        with open(file_path,"rb")as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e,sys)
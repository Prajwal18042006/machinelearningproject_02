# import pandas as pd
# import pickle
# import os


# class CustomData:
#     def __init__(self,
#                  gender,
#                  ethnicity,
#                  parental_level_of_education,
#                  lunch,
#                  test_preparation_course,
#                  reading_score,
#                  writing_score):

#         self.gender = gender.lower()
#         self.ethnicity = ethnicity.lower()  # FIXED
#         self.parental_level_of_education = parental_level_of_education.lower()
#         self.lunch = lunch.lower()
#         self.test_preparation_course = test_preparation_course.lower()

#         self.reading_score = reading_score
#         self.writing_score = writing_score

#     def get_data_as_dataframe(self):
#         payload = {
#             "gender": [self.gender],
#             "race/ethnicity": [self.ethnicity],
#             "parental level of education": [self.parental_level_of_education],
#             "lunch": [self.lunch],
#             "test preparation course": [self.test_preparation_course],
#             "reading score": [self.reading_score],
#             "writing score": [self.writing_score]
#         }

#         return pd.DataFrame(payload)


# class PredictPipeline:
#     def __init__(self):
#         self.model_path = os.path.join("artifacts", "model.pkl")
#         self.preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")

#     def predict(self, features):
#         # Load model
#         with open(self.model_path, "rb") as f:
#             model = pickle.load(f)

#         # Load preprocessor
#         with open(self.preprocessor_path, "rb") as f:
#             preprocessor = pickle.load(f)

#         # Apply transformations
#         data_scaled = preprocessor.transform(features)

#         # Predict
#         preds = model.predict(data_scaled)
#         return preds
import pandas as pd
import pickle
import os


class CustomData:
    def __init__(self,
                 gender,
                 ethnicity,
                 parental_level_of_education,
                 lunch,
                 test_preparation_course,
                 reading_score,
                 writing_score):

        # Must match training categories EXACTLY
        self.gender = gender.lower()

        # FIX FOR ETHNICITY — match: 'group A', 'group B', ...
        eth = ethnicity.strip().lower()
        if eth.startswith("group"):
            # Convert "group a" → "group A"
            self.ethnicity = "group " + eth[-1].upper()
        else:
            self.ethnicity = ethnicity  # fallback just in case

        # FIX parental education (model expects lowercase except apostrophes)
        self.parental_level_of_education = parental_level_of_education.strip().lower()

        # FIX lunch
        self.lunch = lunch.strip().lower()

        # FIX test prep
        self.test_preparation_course = test_preparation_course.strip().lower()

        self.reading_score = reading_score
        self.writing_score = writing_score


    def get_data_as_dataframe(self):
        payload = {
            "gender": [self.gender],
            "race/ethnicity": [self.ethnicity],
            "parental level of education": [self.parental_level_of_education],
            "lunch": [self.lunch],
            "test preparation course": [self.test_preparation_course],
            "reading score": [self.reading_score],
            "writing score": [self.writing_score]
        }

        return pd.DataFrame(payload)


class PredictPipeline:
    def __init__(self):
        self.model_path = os.path.join("artifacts", "model.pkl")
        self.preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")

    def predict(self, features):
        # load model
        with open(self.model_path, "rb") as f:
            model = pickle.load(f)

        # load preprocessor
        with open(self.preprocessor_path, "rb") as f:
            preprocessor = pickle.load(f)

        # 🔥 DEBUG PRINT: show categories the model expects
        print("\n==== ENCODER CATEGORIES (FROM TRAINING) ====")
        try:
            print(preprocessor.named_transformers_["cat_pipeline"]
                  .named_steps["onehot"].categories_)
        except:
            print("Could not print categories. Check transformer names.")
        print("============================================\n")

        # transform
        data_scaled = preprocessor.transform(features)

        # predict
        preds = model.predict(data_scaled)
        return preds

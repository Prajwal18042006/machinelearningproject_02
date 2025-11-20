import os
import sys

# Add src folder to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from flask import Flask, request, render_template
from pipeline.predict_pipeline import CustomData, PredictPipeline

application= Flask(__name__)
app=application


# -------------------------------------------
# 1) DEFAULT PAGE → index.html
# -------------------------------------------
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


# -------------------------------------------
# 2) PREDICTION PAGE → home.html
# -------------------------------------------
@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():

    # When we click the button from index.html
    if request.method == 'GET':
        return render_template('home.html')

    # When the form is submitted in home.html
    data = CustomData(
        gender=request.form.get('gender'),
        ethnicity=request.form.get('ethnicity'),
        parental_level_of_education=request.form.get('parental_level_of_education'),
        lunch=request.form.get('lunch'),
        test_preparation_course=request.form.get('test_preparation_course'),
        reading_score=float(request.form.get('reading_score')),
        writing_score=float(request.form.get('writing_score'))
    )

    pred_df = data.get_data_as_dataframe()
    print(pred_df)

    predict_pipeline = PredictPipeline()
    results = predict_pipeline.predict(pred_df)

    # Show prediction on home.html
    return render_template("home.html", results=results[0])


# -------------------------------------------
# Run App
# -------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)

from flask import Flask, request, render_template
import pandas as pd
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application

@app.route('/')
def index():
    return render_template('index.html', display_results=False)

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html', display_results=False)
    else:
        try:
            # Capture form data
            data = CustomData(
                Gender=request.form.get('Gender'),
                Married=request.form.get('Married'),
                Dependents=int(request.form.get('Dependents')),
                Education=request.form.get('Education'),
                Self_Employed=request.form.get('Self_Employed'),
                ApplicantIncome=int(request.form.get('ApplicantIncome')),
                CoapplicantIncome=int(request.form.get('CoapplicantIncome')),
                LoanAmount=int(request.form.get('LoanAmount')),
                Loan_Amount_Term=int(request.form.get('Loan_Amount_Term')),
                Credit_History=int(request.form.get('Credit_History')),  # Ensure correct type
                Property_Area=request.form.get('Property_Area')
            )

            # Convert to DataFrame
            pref_df = data.get_data_as_data_frame()
            print("Input DataFrame:")
            print(pref_df)

            # Predict
            predict_pipeline = PredictPipeline()
            results = predict_pipeline.predict(pref_df)
            #results_list = results.tolist()  # Convert ndarray to list

            # print("Prediction Results:")
            # print(results_list)

           #threshold = 0.5  # You can adjust this threshold based on your model's behavior
            if results[0] ==1:
                predicted_class = "Approved"  # Approved
            else:
                predicted_class = "Rejected"  # Rejected


            return render_template('home.html', display_results=True, results="Your Request for Loan Amount of ₹ {} is  : {}".format(data.LoanAmount,predicted_class))

        except Exception as e:
            print(f"An error occurred: {e}")
            return render_template('home.html', display_results=False, error=str(e))

if __name__ == '__main__':
    app.run(debug=True,host="0.0.0.0")

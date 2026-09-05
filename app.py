from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# ML model load
model = joblib.load("loan_risk_model.pkl")


@app.route("/", methods=["GET", "POST"])
def home():

    if request.method == "POST":

        # Form se 18 fields lena
        data = {
            "Applicant_Income": float(request.form["Applicant_Income"]),
            "Coapplicant_Income": float(request.form["Coapplicant_Income"]),
            "Employment_Status": request.form["Employment_Status"],
            "Age": int(request.form["Age"]),
            "Marital_Status": request.form["Marital_Status"],
            "Dependents": int(request.form["Dependents"]),
            "Credit_Score": int(request.form["Credit_Score"]),
            "Existing_Loans": int(request.form["Existing_Loans"]),
            "DTI_Ratio": float(request.form["DTI_Ratio"]),
            "Savings": float(request.form["Savings"]),
            "Collateral_Value": float(request.form["Collateral_Value"]),
            "Loan_Amount": float(request.form["Loan_Amount"]),
            "Loan_Term": int(request.form["Loan_Term"]),
            "Loan_Purpose": request.form["Loan_Purpose"],
            "Property_Area": request.form["Property_Area"],
            "Education_Level": request.form["Education_Level"],
            "Gender": request.form["Gender"],
            "Employer_Category": request.form["Employer_Category"]
        }

        # DataFrame banana
        input_df = pd.DataFrame([data])

        # ML prediction
        prediction = model.predict(input_df)[0]

        # Probability nikalna
        probabilities = model.predict_proba(input_df)[0]

        # Yes ki probability
        yes_probability = probabilities[list(model.classes_).index("Yes")] * 100

        # No ki probability
        no_probability = probabilities[list(model.classes_).index("No")] * 100

        print("Prediction:", prediction)
        print("Yes Probability:", yes_probability)
        print("No Probability:", no_probability)

        # Result page par bhejna
        return render_template(
            "result.html",
            prediction=prediction,
            yes_probability=round(yes_probability, 1),
            no_probability=round(no_probability, 1)
        )

    # Home page
    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
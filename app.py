from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load models and transformers
model_multi = joblib.load("air_quality_model_multi.pkl")
poly_multi = joblib.load("poly_transform_multi.pkl")

model_single = joblib.load("air_quality_model_single.pkl")
poly_single = joblib.load("poly_transform_single.pkl")

# Selected features for multi-feature model
selected_features = ["Traffic_Levels", "Temperature_C", "Humidity_%", "Factory_Emissions"]

@app.route("/")
def home():
    return render_template("index.html")  

@app.route("/multi")
def multi():
    return render_template("multi.html")  

@app.route("/single")
def single():
    return render_template("single.html")  

@app.route("/predict_multi", methods=["POST"])
def predict_multi():
    try:
        input_data = {feature: float(request.form[feature]) for feature in selected_features}
        input_df = pd.DataFrame([input_data])  
        input_data_poly = poly_multi.transform(input_df)  
        prediction = model_multi.predict(input_data_poly)[0]

        return render_template("multi.html", prediction=f"Predicted AQI: {prediction:.2f}")
    
    except Exception as e:
        return render_template("multi.html", prediction=f"Error: {str(e)}")

@app.route("/predict_single", methods=["POST"])
def predict_single():
    try:
        factory_emission = float(request.form["Factory_Emissions"])
        input_df = pd.DataFrame([[factory_emission]], columns=["Factory_Emissions"])  
        input_data_poly = poly_single.transform(input_df)  
        prediction = model_single.predict(input_data_poly)[0]

        return render_template("single.html", prediction=f"Predicted AQI: {prediction:.2f}")
    
    except Exception as e:
        return render_template("single.html", prediction=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)

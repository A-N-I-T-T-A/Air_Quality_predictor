import pandas as pd
import joblib
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Load dataset
df = pd.read_csv("air_quality_dataset.csv")

# Define initial features
X = df.drop(columns=["Air_Quality_Index"]) 

y = df["Air_Quality_Index"]

# Add constant term for OLS model
X_ols = sm.add_constant(X)

# Perform Backward Elimination
while True:
    model_ols = sm.OLS(y, X_ols).fit()
    p_values = model_ols.pvalues
    max_p_value = p_values.max()
    if max_p_value > 0.05:  # Remove the least significant feature (p-value > 0.05)
        feature_to_remove = p_values.idxmax()
        if feature_to_remove == "const":  # Ensure we don't remove the constant term
            break
        X_ols = X_ols.drop(columns=[feature_to_remove])
    else:
        break

# Final selected features
selected_features = list(X_ols.columns)
selected_features.remove("const")  # Remove constant from feature list
X = df[selected_features]

print(f"\nSelected features after Backward Elimination: {selected_features}")

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply Polynomial Features Transformation (degree 2)
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# Train Polynomial Regression model
model = LinearRegression()
model.fit(X_train_poly, y_train)

# Predictions
y_pred = model.predict(X_test_poly)

# Evaluate model performance
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Model Accuracy (R² Score): {r2:.4f}")
print(f"Mean Absolute Error: {mae:.2f}")

# Save model & transformer
joblib.dump(model, "air_quality_model_multi.pkl")
joblib.dump(poly, "poly_transform_multi.pkl")



# Train a single-feature polynomial regression model using Factory_Emissions
X_single = df[["Factory_Emissions"]]  # Select only the Factory_Emissions feature
y = df["Air_Quality_Index"]

# Split data into training and testing sets
X_train_single, X_test_single, y_train_single, y_test_single = train_test_split(X_single, y, test_size=0.2, random_state=42)

# Apply Polynomial Features Transformation (degree 2)
poly_single = PolynomialFeatures(degree=2)
X_train_poly_single = poly_single.fit_transform(X_train_single)
X_test_poly_single = poly_single.transform(X_test_single)

# Train Polynomial Regression model
model_single = LinearRegression()
model_single.fit(X_train_poly_single, y_train_single)

# Predictions
y_pred_single = model_single.predict(X_test_poly_single)

# Evaluate model performance
mae_single = mean_absolute_error(y_test_single, y_pred_single)
r2_single = r2_score(y_test_single, y_pred_single)

print(f"Single-Feature Model Accuracy (R² Score): {r2_single:.4f}")
print(f"Single-Feature Mean Absolute Error: {mae_single:.2f}")

# Save single-feature model & transformer
joblib.dump(model_single, "air_quality_model_single.pkl")
joblib.dump(poly_single, "poly_transform_single.pkl")

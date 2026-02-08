import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# ---------------------------
# 1. Data Cleaning Phase
# ---------------------------
# 1.1 Load raw COVID-19 dataset
raw_df = pd.read_csv('owid_covid_data.csv')

# 1.2 Remove unnecessary columns
if 'continent' in raw_df.columns:
    raw_df = raw_df.drop(columns=['continent'])

# 1.3 Handle missing values in key indicators
key_indicators = [
    'total_cases_per_million', 'total_deaths_per_million',
    'median_age', 'aged_65_older',
    'cardiovasc_death_rate', 'diabetes_prevalence',
    'gdp_per_capita', 'human_development_index'
]
raw_df = raw_df.dropna(subset=key_indicators)

# 1.4 Create income group categories based on GDP per capita

def classify_income(gdp):
    if gdp < 1000:
        return 'Low-income'
    elif gdp <= 12000:
        return 'Mid-income'
    else:
        return 'High-income'

raw_df['income_group'] = raw_df['gdp_per_capita'].apply(classify_income)

# 1.5 Save cleaned dataset
cleaned_path = 'cleaned_covid_data.csv'
raw_df.to_csv(cleaned_path, index=False)

# Reload into df for modeling

df = pd.read_csv(cleaned_path)

# ---------------------------
# 2. Machine Learning Pipeline
# ---------------------------
# Q1: Does higher HDI and GDP per capita lead to fewer cases/deaths?
features_q1 = ['gdp_per_capita', 'human_development_index']
target_deaths = 'total_deaths_per_million'
target_cases  = 'total_cases_per_million'

# Split for death-rate model
X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(
    df[features_q1], df[target_deaths], test_size=0.2, random_state=42
)

# Build and evaluate a linear regression pipeline
pipeline_lr = Pipeline([
    ('scaler', StandardScaler()),
    ('lr', LinearRegression())
])

pipeline_lr.fit(X_train_d, y_train_d)
y_pred_d = pipeline_lr.predict(X_test_d)

print("=== Q1: Linear Regression on Death Rate ===")
print("Coefficients:", dict(zip(features_q1, pipeline_lr.named_steps['lr'].coef_)))
print(f"R²: {r2_score(y_test_d, y_pred_d):.3f}")
print(f"MSE: {mean_squared_error(y_test_d, y_pred_d):.3f}\n")

# Repeat for cases per million
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    df[features_q1], df[target_cases], test_size=0.2, random_state=42
)

pipeline_lr.fit(X_train_c, y_train_c)
y_pred_c = pipeline_lr.predict(X_test_c)

print("=== Q1: Linear Regression on Case Rate ===")
print("Coefficients:", dict(zip(features_q1, pipeline_lr.named_steps['lr'].coef_)))
print(f"R²: {r2_score(y_test_c, y_pred_c):.3f}")
print(f"MSE: {mean_squared_error(y_test_c, y_pred_c):.3f}\n")

# Random Forest for death rate feature importance
rf_deaths = RandomForestRegressor(n_estimators=100, random_state=42)
rf_deaths.fit(X_train_d, y_train_d)
imp_q1 = rf_deaths.feature_importances_
print("=== Q1: Random Forest Importances on Death Rate ===")
print(dict(zip(features_q1, imp_q1)))

# Q2: Impact of pre-existing health conditions on mortality
features_q2 = ['cardiovasc_death_rate', 'diabetes_prevalence']
target_mortality = 'total_deaths_per_million'

X_train_h, X_test_h, y_train_h, y_test_h = train_test_split(
    df[features_q2], df[target_mortality], test_size=0.2, random_state=42
)

pipeline_h = Pipeline([
    ('scaler', StandardScaler()),
    ('lr', LinearRegression())
])

pipeline_h.fit(X_train_h, y_train_h)
y_pred_h = pipeline_h.predict(X_test_h)

print("=== Q2: Linear Regression on Health Conditions ===")
print("Coefficients:", dict(zip(features_q2, pipeline_h.named_steps['lr'].coef_)))
print(f"R²: {r2_score(y_test_h, y_pred_h):.3f}")
print(f"MSE: {mean_squared_error(y_test_h, y_pred_h):.3f}\n")

rf_health = RandomForestRegressor(n_estimators=100, random_state=42)
rf_health.fit(X_train_h, y_train_h)
imp_q2 = rf_health.feature_importances_

print("=== Q2: Random Forest Importances on Health Conditions ===")
print(dict(zip(features_q2, imp_q2)))

# Plot feature importances for Q2
plt.figure(figsize=(6,4))
plt.bar(features_q2, imp_q2)
plt.title('Feature Importances: CVD & Diabetes on Mortality')
plt.ylabel('Importance')
plt.tight_layout()
plt.show()

# COVID-19 Data Analysis & Predictive Modeling

## Project Overview

This project analyzes global COVID-19 statistics from the **Our World in Data (OWID)** dataset to uncover relationships between socioeconomic factors, health conditions, and pandemic outcomes. Using machine learning techniques, we explore two primary research questions about COVID-19 mortality and case rates worldwide.

## Project Structure

```
DSproject/
├── project.py                 # Main data processing & ML pipeline
├── Group_21_Final.ipynb       # Comprehensive analysis & visualizations
├── owid-covid-data.csv        # Raw COVID-19 dataset
├── cleaned_covid_data.csv     # Processed dataset (generated)
└── README.md                  # This file
```

## Files Description

### 1. **project.py**
The main Python script that handles:
- **Data Cleaning**: Removes unnecessary columns, handles missing values, creates income classifications
- **Feature Engineering**: Categorizes countries into income groups (Low, Mid, High) based on GDP per capita
- **Machine Learning Models**: 
  - Linear Regression with StandardScaler pipeline
  - Random Forest Regressor for feature importance analysis
- **Model Evaluation**: Computes R² scores and Mean Squared Error (MSE)

### 2. **Group_21_Final.ipynb**
Jupyter notebook containing:
- Detailed exploratory data analysis (EDA)
- Visualizations of key findings
- Model results and interpretations
- Research question discussions

### 3. **owid-covid-data.csv**
Raw dataset from Our World in Data containing:
- COVID-19 statistics (cases, deaths, vaccinations)
- Socioeconomic indicators (GDP, HDI)
- Health metrics (cardiovascular death rate, diabetes prevalence)
- Demographic data (median age, elderly population percentage)

## Research Questions

### **Q1: Economic Development & Pandemic Impact**
*Does higher HDI and GDP per capita lead to fewer COVID-19 cases and deaths?*

**Features Used:**
- GDP per capita
- Human Development Index (HDI)

**Target Variables:**
- Total deaths per million
- Total cases per million

**Models:** Linear Regression, Random Forest

---

### **Q2: Pre-existing Health Conditions & Mortality**
*How do cardiovascular death rates and diabetes prevalence impact COVID-19 mortality?*

**Features Used:**
- Cardiovascular death rate
- Diabetes prevalence

**Target Variable:**
- Total deaths per million

**Models:** Linear Regression, Random Forest

---

## Data Cleaning & Preprocessing

The script performs the following data preparation steps:

1. **Column Removal**: Drops the 'continent' column
2. **Missing Value Handling**: Removes rows with missing values in key indicators:
   - `total_cases_per_million`
   - `total_deaths_per_million`
   - `median_age`
   - `aged_65_older`
   - `cardiovasc_death_rate`
   - `diabetes_prevalence`
   - `gdp_per_capita`
   - `human_development_index`

3. **Feature Engineering**: Creates income group classifications:
   - **Low-income**: GDP < $1,000
   - **Mid-income**: GDP $1,000 - $12,000
   - **High-income**: GDP > $12,000

4. **Output**: Generates `cleaned_covid_data.csv` for modeling

## Machine Learning Pipeline

### Model Architecture
```python
Pipeline:
├── StandardScaler    # Normalize features to zero mean and unit variance
└── LinearRegression  # (or RandomForestRegressor for Q2)
```

### Train-Test Split
- **Training Set**: 80%
- **Test Set**: 20%
- **Random State**: 42 (for reproducibility)

### Evaluation Metrics
- **R² Score**: Explains the proportion of variance in the target variable
- **Mean Squared Error (MSE)**: Measures average squared prediction error

## Installation & Requirements

### Dependencies
```
pandas
scikit-learn
matplotlib
```

### Setup Instructions
1. Install required packages:
   ```bash
   pip install pandas scikit-learn matplotlib
   ```

2. Ensure `owid-covid-data.csv` is in the project directory

3. Run the analysis:
   ```bash
   python project.py
   ```

## Output

Running `project.py` generates:

1. **cleaned_covid_data.csv** - Preprocessed dataset ready for analysis
2. **Console Output** - Model coefficients, R² scores, and MSE values
3. **Feature Importance Plots** - Visualization of which features most impact COVID-19 mortality

### Sample Output Format
```
=== Q1: Linear Regression on Death Rate ===
Coefficients: {'gdp_per_capita': -0.015, 'human_development_index': 45.2}
R²: 0.652
MSE: 1234.567

=== Q1: Random Forest Importances on Death Rate ===
{'gdp_per_capita': 0.35, 'human_development_index': 0.65}
```

## Key Findings (Expected Results)

- **Q1**: Countries with higher GDP and HDI tend to have better pandemic outcomes
- **Q2**: Populations with pre-existing cardiovascular disease and diabetes show higher COVID-19 mortality rates
- **Feature Importance**: HDI generally emerges as a more important predictor than raw GDP in Random Forest models

## How to Use This Project

### For Complete Analysis:
1. Open `Group_21_Final.ipynb` in Jupyter Notebook for full visualizations and discussion
2. Review EDA plots and interpretation of results

### For Data Processing Only:
1. Run `python project.py` to clean data and train models
2. Use the generated `cleaned_covid_data.csv` for further analysis

### For Reproducibility:
- All models use `random_state=42` to ensure consistent results across runs
- StandardScaler ensures comparable coefficients across features

## Future Enhancements

- Include vaccination rates as features
- Implement time-series analysis for temporal COVID-19 trends
- Add geographic clustering to account for regional patterns
- Explore advanced models (XGBoost, Neural Networks)
- Incorporate policy indices or healthcare system rankings

## Dataset Attribution

Data sourced from: **Our World in Data (OWID)**
- Website: https://ourworldindata.org/covid-19
- Last Updated: Continuously updated with latest statistics

## Team

**Group 21** - Data Science Project

---

## Notes

- Missing data handling uses listwise deletion; consider alternative imputation methods for larger-scale analysis
- Linear Regression assumes linear relationships between features and targets
- Random Forest captures non-linear patterns and feature interactions
- Results are specific to the dataset snapshot; real-world conclusions require domain expertise and additional context

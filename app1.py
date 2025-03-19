# Streamlit app setup
import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from concurrent.futures import ThreadPoolExecutor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
# import mlflow
# import mlflow.catboost

st.title("CatBoost Model Optimization with SciPy and MLflow Tracking")
st.write("Optimize features for efficiency using a pre-trained CatBoost model, SciPy optimization, and MLflow tracking.")

# Upload dataset
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    # Load dataset
    data = pd.read_csv(uploaded_file)
    st.write("Dataset Loaded Successfully:")
    st.dataframe(data.head())

    # Load the pre-trained CatBoost model
    st.write("Loading pre-trained CatBoost model...")
    zone = "APAC"  # Example zone
    loaded_model = CatBoostRegressor()
    loaded_model.load_model(rf"C:\Users\40103869\Downloads\model_Arun_zone_APAC (1).cbm")
    st.success(f"Model for zone '{zone}' loaded successfully!")

    # Prepare the dataset for optimization
    features = loaded_model.feature_names_
    data = data[features + ["data", "event_id", "efficiency"]]
    forecast_data = data[data["data"] == "submission_df"]
    # Define variable groups
    # Add event_id filter
    event_ids = forecast_data["event_id"].unique()
    selected_event_id = st.selectbox("Select an Event ID for Optimization", event_ids)

    if selected_event_id:
        forecast_data = forecast_data[forecast_data["event_id"] == selected_event_id]

        # Optimization function
        def objective_function_scipy(vars, other_values, loaded_model, continuous_count, binary_count):
            """
            Objective function for optimization.
            """
            continuous_vars = vars[:continuous_count]
            binary_vars = vars[continuous_count:]
            candidate = list(continuous_vars) + list(binary_vars) + other_values
            return -loaded_model.predict(candidate)

        def scipy_optimization(x_row):
            """
            Optimization using SciPy.
            """
            # Extract problem data
            other_values = list(x_row[other_variables])  # Fixed values for other variables
            continuous_count = len(continuous_variables)
            binary_count = len(binary_variables)

            # Define bounds for variables
            bounds = [(0, 10), (0, 100), (0, 10), (0, 5)] + [(0, 1)] * binary_count

            # Define initial guesses
            x0 = list(x_row[continuous_variables + binary_variables])

            # Define constraints
            constraints = [
                {'type': 'ineq', 'fun': lambda vars: 10 - vars[0]},  # y[0] <= 10
                {'type': 'ineq', 'fun': lambda vars: 100 - vars[1]},  # y[1] <= 100
                {'type': 'ineq', 'fun': lambda vars: 10 - vars[2]},  # y[2] <= 10
                {'type': 'ineq', 'fun': lambda vars: 5 - vars[3]},  # y[3] <= 5
                {'type': 'ineq', 'fun': lambda vars: 1 - vars[continuous_count + 6] - vars[continuous_count + 7]},
            ] + [{'type': 'ineq', 'fun': lambda vars: 1 - vars[continuous_count + i]} for i in range(binary_count)]

            # Solve using SciPy
            result = minimize(
                objective_function_scipy,
                x0=x0,
                args=(other_values, loaded_model, continuous_count, binary_count),
                bounds=bounds,
                constraints=constraints,
                method="COBYLA",
                options={'disp': True}
            )

            # Extract results
            optimized_vars = result.x
            continuous_vars = np.round(optimized_vars[:continuous_count])
            binary_vars = np.round(optimized_vars[continuous_count:])  # Round binary variables

            best_candidate = list(continuous_vars) + list(binary_vars) + other_values
            result_objective = loaded_model.predict(best_candidate)

            return result_objective, list(continuous_vars) + list(binary_vars)

        # Run optimization
        st.write("Running optimizations...")
        with ThreadPoolExecutor() as executor:
            results = list(executor.map(lambda i: scipy_optimization(forecast_data.iloc[i]), range(forecast_data.shape[0])))

        # Prepare results
        efficiencies, optimized_candidates = zip(*results)
        optimized_candidates_df = pd.DataFrame(optimized_candidates, columns=continuous_variables + binary_variables)
        forecast_data["optimized_efficiency"] = efficiencies
        forecast_data["optimized_values"] = optimized_candidates

        # Add a column for identification
        optimized_candidates_df["Type"] = "Optimum"
        forecast_data_subset = forecast_data[continuous_variables + binary_variables].copy()
        forecast_data_subset["Type"] = "Original"

        # Melt and merge for combined table
        optimized_long = pd.melt(
            optimized_candidates_df.reset_index(),
            id_vars=["Index"],
            var_name="Settings",
            value_name="Optimum"
        )
        forecast_long = pd.melt(
            forecast_data_subset.reset_index(),
            id_vars=["Index"],
            var_name="Settings",
            value_name="Original"
        )
        vertical_table = pd.merge(
            optimized_long,
            forecast_long,
            on=["Settings"],
            how="left"
        )
        vertical_table = vertical_table[["Settings", "Optimum", "Original"]]

        # Display and download the table
        st.write("Combined Table (Settings, Optimum, Original):")
        st.dataframe(vertical_table)

        st.download_button(
            label="Download Vertical Table",
            data=vertical_table.to_csv(index=False),
            file_name="vertical_table_settings_optimum_original.csv",
            mime="text/csv",
        )

    else:
        st.warning("No Event ID selected.")
else:
    st.warning("Please upload a CSV file to proceed.")

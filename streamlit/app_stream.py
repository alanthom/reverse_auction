import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from concurrent.futures import ThreadPoolExecutor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
import mlflow
import mlflow.catboost
#  Inject custom CSS to set the background color

import streamlit as st

# Display the uploaded image
import streamlit as st
import base64

# Path to the uploaded image
uploaded_image = r"C:\Users\40103869\Downloads\undefined.png"

# Read and encode the image to Base64
with open(uploaded_image, "rb") as image_file:
    base64_image = base64.b64encode(image_file.read()).decode()

# Add the image to the top-right corner using custom CSS
st.markdown(
    f"""
    <style>
    .top-right-image {{
        position: absolute;
        top: 10px;
        right: 10px;
        width: 100px; /* Adjust size as needed */
    }}
    </style>
    <img src="data:image/png;base64,{base64_image}" class="top-right-image">
    """,
    unsafe_allow_html=True
)



st.markdown(
    """
    <style>
    /* Apply background color to the main content */
    [data-testid="stAppViewContainer"] {
        background-color: #FFD700; /* Golden Yellow */
    }

    /* Customize the sidebar background if needed */
    [data-testid="stSidebar"] {
        background-color: #FFD700; /* Golden Yellow */
    }
    </style>
    """,
    unsafe_allow_html=True
)

def style_dataframe(df):
    return df.style.set_table_styles([
        {
            "selector": "thead th",
            "props": [("background-color", "black"), ("color", "white"), ("font-weight", "bold")]
        },
        {
            "selector": "tbody td",
            "props": [("background-color", "#F5F5F5"), ("color", "black")]
        },
        {
            "selector": "tbody tr:nth-child(even)",
            "props": [("background-color", "#E8E8E8")]
        },
        {
            "selector": "tbody tr:hover",
            "props": [("background-color", "#D3D3D3")]
        },
        {
            "selector": "table",
            "props": [("border-collapse", "collapse"), ("width", "100%")]
        },
        {
            "selector": "th, td",
            "props": [("border", "1px solid #dddddd"), ("padding", "8px")]
        }
    ])

# Streamlit app setup
st.title("Insights Auction Edge")
st.write("Insights for generating combination of rules for best possible efficiency to be confiigured for reverese e-Auctions")

# Initialize MLflow
# mlflow.set_tracking_uri("http://your_mlflow_server:5000")  # Replace with your MLflow server URI
# mlflow.set_experiment("CatBoost Optimization")

# Upload dataset
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    # Load dataset
    data = pd.read_csv(uploaded_file)
    st.write("Dataset Loaded Successfully:")
    # st.dataframe(data.head())

    # Load the pre-trained CatBoost model
    st.write("Loading pre-trained CatBoost model...")
    zone = ["APAC","MAZ","EUR","SAZ","NAZ","AFR"]
    selected_zone_id = st.selectbox("Select a Zone for Optimization", zone)
    if selected_zone_id:
        data = data[data[f"zone_{selected_zone_id}"] == 1]

    category=['COMMERCIAL',"Not Applicable"]
    category_id = st.selectbox("Select L1 category for Optimization", category)

    if category_id != 'Not Applicable':
        data = data[data[f"categ_l1_{category_id}"] == 1]
    else:
        pass

    loaded_model = CatBoostRegressor()

    if category_id !='Not Applicable':
        try:
            loaded_model.load_model(rf"C:\Users\40103869\Anheuser-Busch InBev\Fort Tech - Redemption\model_Arun_zone_{selected_zone_id}_categ_l1_{category_id}.cbm")
            st.success(f"Model for zone '{selected_zone_id}' and '{category_id}' loaded successfully!")
        except:
            st.error(f"Model currently not available!")
    else:
        try:
            loaded_model.load_model(rf"CC:\Users\40103869\Anheuser-Busch InBev\Fort Tech - Redemption\model_Arun_zone_{selected_zone_id} (1).cbm")
            st.error(f"Model for zone '{selected_zone_id}' loaded successfully!")
        except:
            st.success(f"Model currently not available!")
    try:
        # Prepare the dataset for optimization
        features = loaded_model.feature_names_
        data = data[features + ["data", "event_id", "efficiency"]]
        forecast_data = data[data["data"] == "submission_df"]

        # Define variable groups
        binary_variables = [
            'canparticipantsplacebidsduringpreviewperiod_Do not allow prebids',
            'improvebidamountby_Percentage',
            'showlineitemlevelrankinlot_Yes, to Buyers and Participants',
            'enabletrafficlightbidding_Yes',
            'setareviewperiodafterlotoritemcloses_Yes',
            'setareviewperiodafterlotcloses_Yes',
            'canparticipantsseeranks?_Their own rank',
            'canparticipantsseeranks?_No',
        ]

        continuous_variables = [
            'reviewtimeperiod',
            'biddingperiod', 
            'overtimeperiod(minutes)',
            'bidrankthattriggersovertime',
        ]

        other_variables = [
            'baselinespend_usd',
            'Count_supplier',
            'event_template_ABI Reverse Auction - NI',
            'categ_l2_IND CAPEX',
            'item_name',
            'Mean_Lot_Item_Bid_Version',
            'ratio',
            'itemno',
            'bestbid_usd',
            'participant',
            'Count_Item_Name',
        ]

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
            # binary_vars = np.round(vars[continuous_count:])  # Ensure binary variables are rounded to 0 or 1
            candidate = list(continuous_vars) + list(binary_vars) + other_values
            # print(candidate)
            # print(loaded_model.predict(candidate))
            return -loaded_model.predict(candidate)  # Negate because we maximize in Pyomo but SciPy minimizes

        def scipy_optimization(x_row):
            """
            Optimization using SciPy.
            """
            # Extract problem data
            other_values = list(x_row[other_variables])  # Fixed values for other variables
            continuous_count = len(continuous_variables)
            binary_count = len(binary_variables)
            
            # Define bounds for variables
            bounds = [(0, 10), (0,100), (0,10), (0,5)] + [(0, 1)] * binary_count  # Continuous [0, 120], Binary [0, 1]
            
            # Define initial guesses
            x0 = list(x_row[continuous_variables + binary_variables])
            # x0 = [0] * len(continuous_variables + binary_variables)
            # x0 = list(x_row[continuous_variables]) + [0.5] * binary_count

            # Define constraints
            constraints = [
                {'type': 'ineq', 'fun': lambda vars: 10 - vars[0]},  # y[0] <= 10
                {'type': 'ineq', 'fun': lambda vars: 100 - vars[1]},  # y[1] <= 100
                {'type': 'ineq', 'fun': lambda vars: 10 - vars[2]},  # y[2] <= 10
                {'type': 'ineq', 'fun': lambda vars: 5 - vars[3]},  # y[3] <= 5
                {'type': 'ineq', 'fun': lambda vars: 1 - vars[continuous_count + 6] - vars[continuous_count + 7]},  # y[3] <= 5
            ]
            constraints = constraints + [{'type' : 'ineq', 'fun' : lambda vars: 1 - vars[continuous_count + i]} for i in range(binary_count)]

            # Solve using SciPy
            result = minimize(
                objective_function_scipy,
                x0=x0,
                args=(other_values, loaded_model, continuous_count, binary_count),
                bounds=bounds,
                constraints=constraints,
                # method='SLSQP',  # Sequential Least Squares Programming
                method = "COBYLA",
                options={'disp': True}
            )

            # Extract results
            optimized_vars = result.x
            continuous_vars = np.round(optimized_vars[:continuous_count])
            binary_vars = np.round(optimized_vars[continuous_count:])  # Round binary variables

            # Display results
            print("Optimization Result:")
            print("Continuous Variables:", continuous_vars)
            print("Binary Variables:", binary_vars)
            print("initial vars: ", list(x0))

            best_candidate = list(continuous_vars) + list(binary_vars) + other_values
            result_objective = loaded_model.predict(best_candidate) 
            print("Objective Value:", result_objective)

            return result_objective, list(continuous_vars) + list(binary_vars)
        forecast_data= forecast_data.iloc[:1]
        # Run optimization and track with MLflow
        st.write("Running optimizations...")
        # with mlflow.start_run():
        with ThreadPoolExecutor() as executor:
            results = list(executor.map(lambda i: scipy_optimization(forecast_data.iloc[i]), range(forecast_data.shape[0])))

            # Prepare results
            efficiencies, optimized_candidates = zip(*results)
            optimized_candidates_df = pd.DataFrame(optimized_candidates,columns=continuous_variables+binary_variables)
            forecast_data["optimized_efficiency"] = efficiencies
            forecast_data["optimized_values"] = optimized_candidates

            # Log metrics to MLflow
            avg_efficiency = np.mean(efficiencies)
            # st.write(f"Average Optimized Efficiency: {avg_efficiency}")
            # mlflow.log_metric("avg_optimized_efficiency", avg_efficiency)

            # Log parameters
            # mlflow.log_param("zone", zone)
            # mlflow.log_param("continuous_variables_count", len(continuous_variables))
            # mlflow.log_param("binary_variables_count", len(binary_variables))

            # Log optimized results
            results_file = "optimized_results.csv"
            forecast_data.to_csv(results_file, index=False)
            # mlflow.log_artifact(results_file)
        
            st.success("Optimization completed!")

        # Display results
        y_pred = loaded_model.predict(forecast_data[continuous_variables+binary_variables+other_variables])

        # Combine y_pred and forecast_data into a single DataFrame
        combined_df = pd.DataFrame({
            "Original Efficiency": y_pred.flatten(),  # Ensure y_pred is flattened if it's multidimensional
            "Optimized Efficiency": forecast_data["optimized_efficiency"],
            
        })
        # Calculate uplift and uplift dollars
        # combined_df["uplift"] = combined_df["Optimized Efficiency"] - combined_df["Original Efficiency"]
        # combined_df['Uplift_dollars'] = combined_df['uplift'].astype(float) * forecast_data['bestbid_usd'].astype(float) * forecast_data['item_name'].astype(float)

        # # Add arrow indicators to the uplift column
        # def add_arrows(value):
        #     if value > 0:
        #         return f"▲ {np.round(value, 5)}"  # Positive uplift
        #     elif value < 0:
        #         return f"▼ {np.round(value, 5)}"  # Negative uplift
        #     else:
        #         return f"— {np.round(value, 5)}"  # No change

        # # Create a formatted version of the uplift column with arrows
        # combined_df["uplift"] = combined_df["uplift"].apply(add_arrows)

        # # Display the combined DataFrame
        # st.write("Efficiency Comparison:")
        # styled_df = combined_df[["Original Efficiency", "Optimized Efficiency", "uplift", "Uplift_dollars"]]
        # st.dataframe(styled_df)

        combined_df["uplift"] = combined_df["Optimized Efficiency"] - combined_df["Original Efficiency"]
        combined_df['Uplift_dollars'] = combined_df['uplift'].astype(float) * forecast_data['bestbid_usd'].astype(float) * forecast_data['item_name'].astype(float)

        # Add arrow indicators and convert uplift into percentage
        def add_arrows_as_percentage(value):
            percentage_value = value * 100  # Convert to percentage
            if percentage_value > 0:
                return f"▲ {np.round(percentage_value, 2)}%"  # Positive uplift with %
            elif percentage_value < 0:
                return f"▼ {np.round(percentage_value, 2)}%"  # Negative uplift with %
            else:
                return f"— {np.round(percentage_value, 2)}%"  # No change

        # Apply the function to the uplift column
        combined_df["uplift"] = combined_df["uplift"].apply(add_arrows_as_percentage)

        # Display the combined DataFrame
        st.write("Efficiency Comparison:")
        styled_df = combined_df[["Original Efficiency", "Optimized Efficiency", "uplift", "Uplift_dollars"]]
        # Convert "Original Efficiency" and "Optimized Efficiency" columns to percentage format
        styled_df["Original Efficiency"] = (styled_df["Original Efficiency"] * 100).apply(lambda x: f"{x:.2f}%")
        styled_df["Optimized Efficiency"] = (styled_df["Optimized Efficiency"] * 100).apply(lambda x: f"{x:.2f}%")
        styled_df=styled_df.reset_index(drop=True)
        
        st.dataframe(styled_df)



        # Display the combined DataFrame
        # st.write("Efficiency Comparison:")
        # st.dataframe(style_dataframe(combined_df),use_container_width=True)
        # # Combine the two dataframes into a vertical table
        optimized_candidates_df["Source"] = "Optimized Candidates"
        forecast_data_subset = forecast_data[continuous_variables + binary_variables].copy()
        forecast_data_subset["Source"] = "Forecast Data"

        # # Concatenate the two dataframes
        # vertical_table = pd.concat([optimized_candidates_df, forecast_data_subset], axis=0, ignore_index=True)

        # # Display the vertical table
        # st.write("Combined Vertical Table:")
        # st.dataframe(vertical_table)

        # # Download the vertical table
        # st.download_button(
        #     label="Download Vertical Table",
        #     data=vertical_table.to_csv(index=False),
        #     file_name="vertical_table.csv",
        #     mime="text/csv",
        # )

        # Add a column for identification
        # optimized_candidates_df["Type"] = "Optimum"
        # forecast_data_subset["Type"] = "Original"

        # Rename the index to match for merging
        optimized_candidates_df.index.name = "Index"
        forecast_data_subset.index.name = "Index"

        # Melt both dataframes to long format
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
        

        # Merge the long-format dataframes on 'Settings' and 'Index'
        vertical_table = pd.merge(
            optimized_long,
            forecast_long,
            on=["Settings"],
            how="left"
        )  # Drop index column if not needed

        # Rearrange columns to match the desired format
        vertical_table = vertical_table[["Settings", "Original","Optimum"]]
        # Display the vertical table
        st.write("Rules Compared:")
        # st.dataframe(vertical_table)
        st.dataframe(style_dataframe(vertical_table),use_container_width=True)

        # Download the vertical table
        st.download_button(
            label="Download Recommendation",
            data=vertical_table.to_csv(index=False),
            file_name="vertical_table_settings_optimum_original.csv",
            mime="text/csv",
        )
        # final = pd.DataFrame(optimized_candidates, columns = loaded_model.feature_names_[:len(continuous_variables) + len(binary_variables)+len(other_variables)])
    except:
        st.warning("Service not available at the moment")        
else:
    st.warning("Please upload a CSV file to proceed.")


# English Reverse Auction Framework for Procurement

This repository provides an **English Reverse Auction** framework designed to help companies procure supplies from vendors efficiently. The system leverages machine learning (CatBoost algorithm) and optimization techniques (COBYLA) to model historical auction data and optimize auction settings, driving the bid down to the best possible price point for the company.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Technologies](#technologies)
- [How It Works](#how-it-works)
- [Installation](#installation)
- [Usage](#usage)
- [Streamlit Optimization App](#streamlit-optimization-app)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Overview

The goal of this framework is to automate and optimize the reverse auction process for procurement by utilizing historical bid data to predict optimal auction configurations. The framework includes:

- **CatBoost Model:** A machine learning model to analyze past auction data and predict bidding behavior.
- **COBYLA Optimizer:** An optimization algorithm to fine-tune auction settings, ensuring the best price outcome for the company.
- **Streamlit-Based Optimization App:** A user-friendly web app for planners to design auction settings based on the optimizer's recommendations.

This project was developed as part of a **hackathon** and won an award for its innovative approach to procurement.

## Features

- Predicts optimal bidding behavior using historical auction data.
- Utilizes **CatBoost**, a powerful gradient boosting algorithm for efficient prediction.
- Applies **COBYLA** optimization to fine-tune auction settings and drive down the bid price.
- Interactive **Streamlit app** that allows planners to visualize and set auction configurations based on model predictions.

## Technologies

- **CatBoost**: A gradient boosting algorithm used for the machine learning model.
- **COBYLA**: Constrained optimization BY Linear Approximations for optimization of auction settings.
- **Streamlit**: Web framework to create the interactive auction optimization app.
- **Python**: Core programming language for model development and optimization.

## How It Works

1. **Data Preprocessing:**
   - Historical bid data is preprocessed and anonymized for model training.
   - The data is masked to ensure no sensitive information is exposed.

2. **CatBoost Model:**
   - The CatBoost algorithm is trained on historical auction data to learn bidding patterns and predict the optimal auction settings.

3. **COBYLA Optimization:**
   - The optimizer, COBYLA, takes the output from the CatBoost model and applies constraints to fine-tune the auction configuration.
   - The goal is to drive the bid price to the most favorable point for the company while maintaining fairness in the bidding process.

4. **Streamlit Optimization App:**
   - A Streamlit app allows planners to interact with the system, input auction parameters, and receive recommendations on optimal settings based on the optimization model.
   - The app helps planners visualize the auction configurations and adjust settings as needed.

## Installation

To set up the repository locally, follow the instructions below:

### Prerequisites

Make sure you have Python 3.7+ installed on your system.

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/repository-name.git
   cd repository-name
   ```

2. Create a virtual environment (optional but recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate   # For Linux/Mac
   venv\Scripts\activate      # For Windows
   ```

3. Install required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Run the Streamlit app:

   ```bash
   streamlit run app.py
   ```

   This will launch the Streamlit app in your browser where you can interact with the optimization model.

## Usage

After launching the Streamlit app, you can use the interface to:

1. **Input Auction Settings:** Define parameters such as auction type, initial bid, and time limits.
2. **View Recommendations:** The optimizer will suggest optimal auction configurations based on historical data.
3. **Adjust Settings:** Modify the auction parameters and rerun the optimization as needed.

## Streamlit Optimization App

The Streamlit-based app provides a simple interface for planners to visualize and fine-tune auction settings. The app offers:

- **Auction Configuration Interface**: A form where planners can input and adjust auction parameters.
- **Optimization Results**: Displays the recommended auction settings based on the optimizer’s output.
- **Interactive Graphs and Plots**: Visualize bidding patterns, price points, and optimization results.

## Contributing

We welcome contributions to enhance and extend this project! If you'd like to contribute:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes.
4. Commit your changes (`git commit -am 'Add new feature'`).
5. Push to your fork (`git push origin feature-branch`).
6. Open a pull request.

Please ensure that your code follows the project's coding style and includes tests if applicable.

## Acknowledgments

- Thanks to the organizers of the hackathon for providing the opportunity to develop this project.
- Special thanks to the open-source contributors whose libraries and frameworks were used in this project.
- CatBoost and COBYLA for providing the powerful tools for machine learning and optimization.
- Thanks to my FORGE Team

---

If you have any questions or need further assistance, feel free to open an issue or reach out to us directly!

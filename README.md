# CarVal AI: Car Price Prediction with Machine Learning

An end-to-end Machine Learning web application designed to predict used car prices. The application is trained on the real Kaggle **Used Cars Dataset** (originally sourced from CarDekho) and implements a fully client-side Machine Learning model execution engine.

---

## Key Features

1. **Real-World Dataset Integration**: Built and evaluated using the authentic 301-record Kaggle used cars dataset containing variables like manufacturing year, original ex-showroom price, total distance driven, fuel types, seller types, transmissions, and previous owners.
2. **Robust Multi-Model Regression Pipeline**:
   - Automatically compares three regression algorithms: **Linear Regression**, **Random Forest**, and **Gradient Boosting**.
   - Selects the best performing model based on the highest $R^2$ score (Gradient Boosting achieves **$R^2 \approx 0.97$**).
3. **Dynamic Client-Side ML Execution**:
   - The best trained model's parameters (e.g., decision trees, thresholds, leaf values) are exported to `public/meta.json` during the training run.
   - Predictions are computed **locally in the browser** using a custom tree-walking algorithm in JavaScript, enabling serverless and database-less deployment.
4. **Interactive Dashboard & Visualization UI**:
   - **Dynamic Metrics**: Loads R² score, Mean Absolute Error (MAE), and RMSE dynamically from training outputs.
   - **Feature Importance**: Visualizes the decision weighting of each feature dynamically.
   - **Rich Visualizations**: Serves Matplotlib/Seaborn analytics plots directly in the UI tabs:
     * *Actual vs. Predicted Plot*
     * *Price Distribution Histogram*
     * *Average Price by Car Name*
     * *Model Performance Evaluation Comparison*
   - **Modern Aesthetic**: Built with a dark mode glassmorphism UI using Google Fonts (Inter) and custom CSS.

---

## Directory Structure

```
├── api/
│   ├── model.pkl               # Pickled best-performing ML model (Gradient Boosting)
│   └── scaler.pkl              # Scaler object for numerical features
├── public/
│   ├── index.html              # Main dashboard frontend interface
│   ├── meta.json               # Exported model parameters, performance metrics, and encoders
│   ├── chart_actual_vs_predicted.png
│   ├── chart_brand_price.png
│   ├── chart_feature_importance.png
│   ├── chart_model_comparison.png
│   └── chart_price_distribution.png
├── car_data.csv                # Real Kaggle used cars dataset
├── train_model.py              # ML training, evaluation, parameter exporter, and plotting script
└── README.md                   # Project documentation
```

---

## Setup & Running Guide

### 1. Install Dependencies
Ensure you have Python installed, then install the necessary scientific packages:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib
```

### 2. Train the Model & Export Metadata
To re-run the training pipeline, evaluate the regression models, and export the metadata:
```bash
python train_model.py
```
This script will update `api/model.pkl`, export target metrics and tree parameters to `public/meta.json`, and refresh the Matplotlib charts.

### 3. Launch the Web Application
Serve the `public` directory using a local HTTP server:

* **Using Python:**
  ```bash
  python -m http.server 8000 --directory public
  ```
* **Using Node.js (`serve`):**
  ```bash
  npx serve public
  ```

Open **`http://localhost:8000`** (or the port specified by Node) in your web browser.

---

## Feature Columns Used for Predictions

* **Car Name**: Dropdown containing the unique vehicle models from the dataset (e.g., Ciaz, Fortuner, City, etc.).
* **Present Price**: Ex-showroom showroom price of the car (in Lakhs INR).
* **Year**: The year of manufacturing.
* **Kms Driven**: Total distance the car has traveled.
* **Fuel Type**: Fuel category (Petrol, Diesel, CNG).
* **Seller Type**: Seller category (Dealer, Individual).
* **Transmission**: Transmission category (Manual, Automatic).
* **Owner**: The number of previous owners (0, 1, 2, or 3).

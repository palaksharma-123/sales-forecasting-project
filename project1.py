import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

# --- 1. Generate Dummy CSV Data ---
# If you have your own sales.csv, you can skip this function and directly load it.
def generate_dummy_sales_data(filename="sales.csv", num_entries=365):
    """
    Generates a dummy CSV file with sales data.
    """
    start_date = datetime(2023, 1, 1)
    products = ['Product_A', 'Product_B', 'Product_C', 'Product_D']
    data = []

    for i in range(num_entries):
        current_date = start_date + timedelta(days=i)
        for product in products:
            quantity = np.random.randint(10, 100)
            revenue = quantity * np.random.uniform(5.0, 25.0) # Price per unit varies
            data.append([current_date.strftime('%Y-%m-%d'), product, quantity, round(revenue, 2)])

    df = pd.DataFrame(data, columns=['date', 'product', 'quantity', 'revenue'])
    df.to_csv(filename, index=False)
    print(f"Dummy sales data saved to {filename}")

# Generate the dummy data (run this once if you don't have a CSV)
generate_dummy_sales_data("sales.csv", num_entries=730) # Generating 2 years of data

# --- 2. Load Data ---
try:
    df = pd.read_csv("sales.csv")
    print("Data loaded successfully:")
    print(df.head())
    print(df.info())
except FileNotFoundError:
    print("Error: sales.csv not found. Please ensure the file exists or run generate_dummy_sales_data().")
    exit()

# --- 3. Preprocess Data ---

# Convert 'date' column to datetime objects
df['date'] = pd.to_datetime(df['date'])

# Handle potential null values (example: fill with median for numerical, mode for categorical)
# For 'quantity' and 'revenue', fill with median
df['quantity'].fillna(df['quantity'].median(), inplace=True)
df['revenue'].fillna(df['revenue'].median(), inplace=True)

# For 'product', fill with mode (most frequent product)
df['product'].fillna(df['product'].mode()[0], inplace=True)

print("\nNull values after handling:")
print(df.isnull().sum())

# Feature Engineering: Convert date to a numerical format (e.g., ordinal, or timestamp)
# Using ordinal date as a simple numerical representation for linear regression
df['date_ordinal'] = df['date'].apply(lambda date: date.toordinal())

# One-hot encode the 'product' categorical column
df = pd.get_dummies(df, columns=['product'], drop_first=True) # drop_first avoids multicollinearity

print("\nDataFrame after preprocessing and feature engineering:")
print(df.head())

# Define features (X) and target (y)
# We'll use 'date_ordinal' and one-hot encoded product columns as features to predict 'revenue'.
# 'quantity' can also be a feature if you want to predict revenue based on quantity sold and date/product.
# For simplicity, let's predict revenue based on date and product.
X = df[['date_ordinal'] + [col for col in df.columns if 'product_' in col]]
y = df['revenue']

# --- 4. Apply Linear Regression ---

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nModel Evaluation:")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R-squared (R2): {r2:.2f}")

# --- 5. Plot Actual vs. Predicted Sales ---

plt.figure(figsize=(12, 6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.7)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2) # Line for perfect prediction
plt.title('Actual vs. Predicted Sales Revenue')
plt.xlabel('Actual Sales Revenue')
plt.ylabel('Predicted Sales Revenue')
plt.grid(True)
plt.show()

# Visualize predictions over time for a subset to see trends
# Sort the test set by date for better visualization
X_test_sorted = X_test.copy()
X_test_sorted['predicted_revenue'] = y_pred
X_test_sorted['actual_revenue'] = y_test

# Convert ordinal back to date for plotting
X_test_sorted['date'] = X_test_sorted['date_ordinal'].apply(lambda ordinal: datetime.fromordinal(ordinal))
X_test_sorted = X_test_sorted.sort_values(by='date')

plt.figure(figsize=(14, 7))
sns.lineplot(x='date', y='actual_revenue', data=X_test_sorted.head(100), label='Actual Revenue', marker='o', markersize=5)
sns.lineplot(x='date', y='predicted_revenue', data=X_test_sorted.head(100), label='Predicted Revenue', marker='x', markersize=5)
plt.title('Actual vs. Predicted Sales Revenue Over Time (First 100 Test Points)')
plt.xlabel('Date')
plt.ylabel('Sales Revenue')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# --- 6. Forecast for Upcoming Sales Periods ---

# Define the period for forecasting (e.g., next 30 days)
forecast_start_date = df['date'].max() + timedelta(days=1)
forecast_end_date = forecast_start_date + timedelta(days=30) # Forecast for next 30 days

# Create future dates for forecasting
future_dates = pd.date_range(start=forecast_start_date, end=forecast_end_date, freq='D')

# Create a DataFrame for future predictions
future_df = pd.DataFrame({'date': future_dates})
future_df['date_ordinal'] = future_df['date'].apply(lambda date: date.toordinal())

# Replicate the one-hot encoding structure for future predictions
# Assuming all products might be sold in the future, or you can specify which products
# For simplicity, let's create forecasts for a specific product (e.g., Product_A)
# You would need to iterate or create rows for each product you want to forecast.
# Let's forecast for 'Product_A' as an example.
# Create a dummy row for each product you want to forecast
forecast_products_data = []
for f_date in future_dates:
    for product in products: # Using the 'products' list from dummy data generation
        row = {'date': f_date, 'date_ordinal': f_date.toordinal()}
        # Initialize all product columns to 0
        for p_col in [col for col in df.columns if 'product_' in col]:
            row[p_col] = 0
        # Set the current product's one-hot column to 1
        if f'product_{product}' in row: # Ensure the column exists from training data
            row[f'product_{product}'] = 1
        forecast_products_data.append(row)

forecast_df_products = pd.DataFrame(forecast_products_data)

# Ensure all product columns from the training set are present in the forecast_df_products
# and fill missing ones with 0 (for products not being forecasted in a specific row)
for col in [c for c in X.columns if 'product_' in c]:
    if col not in forecast_df_products.columns:
        forecast_df_products[col] = 0

# Select only the feature columns present in X (training data) for prediction
X_forecast = forecast_df_products[[col for col in X.columns if col in forecast_df_products.columns]]

# Make predictions for the future period
forecast_df_products['predicted_revenue'] = model.predict(X_forecast)

# Add the actual product name back for clarity
def get_product_name(row):
    for col in [c for c in forecast_df_products.columns if 'product_' in c]:
        if row[col] == 1:
            return col.replace('product_', '')
    return 'Unknown' # Should not happen if logic is correct

forecast_df_products['product'] = forecast_df_products.apply(get_product_name, axis=1)

print("\n--- Sales Forecast for Upcoming Periods (Next 30 Days) ---")
# Display the forecast (date, product, predicted revenue)
final_forecast = forecast_df_products[['date', 'product', 'predicted_revenue']].round(2)
print(final_forecast.head(10)) # Display first 10 entries of the forecast

# Group by date to get total predicted revenue per day for plotting
daily_forecast = final_forecast.groupby('date')['predicted_revenue'].sum().reset_index()

plt.figure(figsize=(14, 7))
sns.lineplot(x='date', y='predicted_revenue', data=daily_forecast, marker='o')
plt.title('Total Predicted Sales Revenue for Upcoming 30 Days')
plt.xlabel('Date')
plt.ylabel('Predicted Sales Revenue')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Save forecast to CSV
final_forecast.to_csv("sales_forecast.csv", index=False)
print("\nSales forecast saved to sales_forecast.csv")

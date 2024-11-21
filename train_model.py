import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from linear_regression import LinearRegressionScratch
from kmeans import KMeansScratch
import joblib
import numpy as np

# Load the dataset
df = pd.read_csv('house_prices.csv')

# Standardize location names to lowercase
df['location'] = df['location'].str.lower()

# Convert size to numerical values using the provided function (from app.py)
def convert_size(size):
    if isinstance(size, str):
        size_parts = size.split('-')
        if len(size_parts) == 4:
            return int(size_parts[0]) * 5476 + int(size_parts[1]) * 342.25 + int(size_parts[2]) * 342.25 + int(size_parts[3]) * 85.56
        elif 'Aana' in size:
            return float(size.replace(' Aana', '')) * 342.25
        else:
            return float(size)
    return size

df['size'] = df['size'].apply(convert_size)

y = df['price'].values

# One-Hot Encode the location column
onehot_encoder = OneHotEncoder(sparse_output=False)
encoded_locations = onehot_encoder.fit_transform(df[['location']])

# Combine the encoded locations with the other features
X = np.hstack([encoded_locations, df[['size', 'bedrooms', 'bathroom', 'floor', 'year_of_construction', 'ft_road']].values])

# Apply feature scaling to the features (excluding location since it's already encoded)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply KMeans clustering to scaled data
kmeans = KMeansScratch(n_clusters=5)  # Adjust the number of clusters as needed
kmeans.fit(X_scaled)

# Add cluster labels as a feature
df['cluster'] = kmeans.labels
df.to_csv('house_prices.csv', index=False)

# Combine scaled features and cluster labels
X_with_cluster = np.hstack([X_scaled, df[['cluster']].values])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_with_cluster, y, test_size=0.1, random_state=42)

# Train model using custom Linear Regression implementation
model = LinearRegressionScratch()
model.fit(X_train, y_train, learning_rate=0.01, epochs=5000)

# Inspect weights and bias
weights, bias = model.get_weights()
print("Weights:", weights)
print("Bias:", bias)

# Predict on test set
y_pred = model.predict(X_test)

# Compute R-squared
ss_total = np.sum((y_test - np.mean(y_test)) ** 2)
ss_residual = np.sum((y_test - y_pred) ** 2)
r_squared = 1 - (ss_residual / ss_total)

# Convert R-squared to percentage
r_squared_percentage = r_squared * 100

# Save model, KMeans model, scaler, and R-squared
joblib.dump(model, 'trained_model.pkl')
joblib.dump(kmeans, 'trained_kmeans.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(r_squared_percentage, 'model_r_squared_percentage.pkl')
# Save the encoder for later use during prediction
joblib.dump(onehot_encoder, 'onehot_encoder.pkl')

print(f"Model saved successfully. R-squared: {r_squared_percentage:.2f}%")

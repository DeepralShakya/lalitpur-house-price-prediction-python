import matplotlib
from flask import Flask, request, jsonify, send_file
import joblib
from flask_cors import CORS
import numpy as np
import pandas as pd
import datetime
import uuid
import matplotlib.pyplot as plt
import seaborn as sns  # Import seaborn for better visualization

matplotlib.use('Agg')  # Use Agg backend for non-interactive plotting


# Load the trained model, scaler, kmeans, and onehot_encoder
model = joblib.load('trained_model.pkl')
scaler = joblib.load('scaler.pkl')
kmeans = joblib.load('trained_kmeans.pkl')
r_squared_percentage = joblib.load('model_r_squared_percentage.pkl')
onehot_encoder = joblib.load('onehot_encoder.pkl')

app = Flask(__name__)
CORS(app)

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

def format_to_nepali_style(number):
    str_number = str(int(number))
    last_three_digits = str_number[-3:]
    rest_of_the_number = str_number[:-3]

    if not rest_of_the_number:
        return last_three_digits

    rest_of_the_number_formatted = ''
    while len(rest_of_the_number) > 0:
        rest_of_the_number_formatted = ',' + rest_of_the_number[-2:] + rest_of_the_number_formatted
        rest_of_the_number = rest_of_the_number[:-2]

    return rest_of_the_number_formatted[1:] + ',' + last_three_digits

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    try:
        location = data['location'].lower().strip()
        size = convert_size(data['size'])
        bedrooms = int(data['bedrooms'])
        bathroom = float(data['bathroom'])
        floor = float(data['floor'])
        year_of_construction = int(data['year_of_construction'])
        ft_road = int(data['ft_road'])

        # Validate year_of_construction (no changes needed here)
        current_year = datetime.datetime.now().year + 57
        warning = None
        if year_of_construction > current_year:
            warning = f'Prediction for future years (beyond {current_year}) may not be accurate.'

        # Check if the location is in the onehot_encoder categories
        known_locations = onehot_encoder.categories_[0]
        if location not in known_locations:
            return jsonify({'error': f"Location '{location}' not found in the dataset."}), 400

        # Encode the location
        input_location_df = pd.DataFrame({'location': [location]})
        encoded_location = onehot_encoder.transform(input_location_df)

        # Prepare data for prediction
        input_data = np.array([[size, bedrooms, bathroom, floor, year_of_construction, ft_road]])

        # Concatenate the encoded location with the other input data
        input_data_with_location = np.hstack([encoded_location, input_data])

        # Apply scaling using the same scaler from training
        input_data_scaled = scaler.transform(input_data_with_location)

        # Apply KMeans clustering to the scaled input data to get cluster labels
        cluster_label = kmeans.predict(input_data_scaled)

        # Reshape cluster_label to 2D array (shape (1, 1)) for concatenation
        cluster_label = cluster_label.reshape(-1, 1)

        # Add the cluster label as a feature
        input_data_with_cluster = np.hstack([input_data_scaled, cluster_label])

        # Predict using the model
        prediction = model.predict(input_data_with_cluster)[0]

        # Check for negative prediction
        if prediction < 0:
            prediction = 200000
            year_difference = year_of_construction - 2060
            prediction += year_difference * 10000
            if size > 1:
                prediction += (size - 1) * 60000
            if bedrooms > 1:
                prediction += (bedrooms - 1) * 40000
            if bathroom > 1:
                prediction += (bathroom - 1) * 30000
            if floor > 1:
                prediction += (floor - 1) * 20000

        if np.isnan(prediction):
            return jsonify({'error': 'Prediction resulted in NaN value. Check input data or model.'}), 400

        formatted_price = format_to_nepali_style(prediction)

        response = {
            'predicted_price': formatted_price,
            'r_squared': f'{r_squared_percentage:.2f}',
        }

        if warning:
            response['warning'] = warning

        return jsonify(response)
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/cluster-analysis', methods=['GET'])
def cluster_analysis():
    try:
        # Load the dataset
        df = pd.read_csv('house_prices.csv')

        # Get the 'x_feature' from the request arguments, defaulting to 'size' if not provided
        x_feature = request.args.get('x_feature', 'size')

        # Check if the provided feature is in the dataset
        if x_feature not in df.columns:
            return jsonify({'error': f'Feature "{x_feature}" not found in dataset.'}), 400

        # Create a unique filename for the plot
        unique_filename = f'static/cluster_analysis_{uuid.uuid4()}.png'

        # Create the plot
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=df, x=x_feature, y='price', hue='cluster', palette='viridis', style='cluster', markers='o', s=100)
        plt.title(f'Cluster Analysis of House Prices based on {x_feature.capitalize()}')
        plt.xlabel(x_feature.capitalize())
        plt.ylabel('Price')

        # Disable scientific notation on the y-axis
        plt.ticklabel_format(style='plain', axis='y')

        plt.legend(title='Cluster')
        plt.grid(True)

        # Save the plot with the unique filename
        plt.savefig(unique_filename)
        plt.close()  # Close the plot to free memory

        # Return the URL of the saved image
        return jsonify({'graph': f'/{unique_filename}'})
    except Exception as e:
        return jsonify({'error': str(e)}), 400



if __name__ == '__main__':
    app.run(debug=True)

import os  # For accessing environment variables
import librosa  # For audio processing (loading and feature extraction)
import numpy as np  # For numerical operations and array handling
import pandas as pd  # For reading the pickled DataFrame
from sklearn.preprocessing import LabelEncoder  # For encoding species labels
from tensorflow.keras.models import load_model  # For loading the pre-trained ANN model
from flask import Flask, render_template, request  # For building the web app

# Initialize Flask application
app = Flask(__name__)

# Load preprocessed data and set up label encoding
final = pd.read_pickle("extracted_df.pkl")  # Load pickled DataFrame with audio features and labels
y = np.array(final["name"].tolist())  # Extract species names as a NumPy array
le = LabelEncoder()  # Initialize label encoder for species names
le.fit_transform(y)  # Fit the encoder to species labels (note: transformed result not stored)
Model1_ANN = load_model("Model1.h5")  # Load the pre-trained ANN model from file

# Function to extract MFCC features from an audio file
def extract_feature(audio_path):
    """Extract MFCC features from an audio file for model prediction."""
    audio_data, sample_rate = librosa.load(audio_path, res_type="kaiser_fast")  # Load audio with fast resampling
    feature = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=40)  # Compute 40 MFCCs
    feature_scaled = np.mean(feature.T, axis=0)  # Average across time axis for fixed-size input
    return np.array([feature_scaled])  # Return as a 2D array (1 sample, feature vector)

# Function to predict bird species from audio
def ANN_print_prediction(audio_path):
    """Predict bird species using the ANN model and return the class name."""
    prediction_feature = extract_feature(audio_path)  # Extract features from the audio
    predicted_vector = np.argmax(Model1_ANN.predict(prediction_feature), axis=-1)  # Get predicted class index
    predicted_class = le.inverse_transform(predicted_vector)  # Convert index back to species name
    return predicted_class[0]  # Return the predicted species as a string

# Route for the homepage
@app.route("/")
@app.route("/index")
def index():
    """Render the index (homepage) template."""
    return render_template('index.html')

# Route for the upload page (GET request only)
@app.route("/upload", methods=['GET'])
def upload():
    """Render the upload page template for audio file submission."""
    return render_template("upload.html")

# Route for the chart page
@app.route("/chart")
def chart():
    """Render the chart page template (e.g., for visualization)."""
    return render_template('chart.html')

# Route for the about page
@app.route("/about")
def about():
    """Render the about page template."""
    return render_template('about.html')


# Route to handle audio file submission and prediction
@app.route("/submit", methods=['GET', 'POST'])
def get_output():
    """Handle audio file upload, predict species, and render result."""
    if request.method == 'POST':  # Check if the request is a POST (file submission)
        audio_path = request.files['wavfile']  # Get the uploaded audio file from the form
        img_path = "static/tests/" + audio_path.filename  # Define save path (named img_path but for audio)
        audio_path.save(img_path)  # Save the uploaded file to the static folder
        predict_result = ANN_print_prediction(img_path)  # Predict species from the audio file
    return render_template("prediction.html", prediction=predict_result, audio_path=img_path)  # Render result page

# Main entry point for running the Flask app
if __name__ == '__main__':
    """Run the Flask app with specified host and port."""
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))  # Run on all interfaces, default port 8080

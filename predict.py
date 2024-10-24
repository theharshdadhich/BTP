import numpy as np
import pandas as pd
from google.colab import files

# Function to generate synthetic dataset
def generate_synthetic_data(num_samples):
    data = []
    for _ in range(num_samples):
        planet_position = np.random.randint(0, 360)  # Random degree between 0 and 360
        zodiac_index = np.random.randint(0, 12)  # Random zodiac sign index (0 to 11 for 12 signs)
        career_success = np.random.choice([0, 1])  # Random career outcome (0 = Failure, 1 = Success)

        # One-hot encoding for zodiac signs (12 zodiac signs)
        zodiac_encoding = [0] * 12
        zodiac_encoding[zodiac_index] = 1

        # Combine all features into a row (planet position + one-hot zodiac encoding + career outcome)
        row = [planet_position] + zodiac_encoding + [career_success]
        data.append(row)

    # Create DataFrame with appropriate column names
    columns = ['position'] + [f'zodiac_{i}' for i in range(12)] + ['career_success']
    return pd.DataFrame(data, columns=columns)

# Generate a dataset with 100 samples
synthetic_df = generate_synthetic_data(100)

# Save the generated dataset to a CSV file
synthetic_df.to_csv('synthetic_astrology_data.csv', index=False)
print("Dataset generated and saved as 'synthetic_astrology_data.csv'.")

# Display the first few rows of the dataset
print(synthetic_df.head())

# If using Google Colab, download the generated CSV file
try:
    files.download('synthetic_astrology_data.csv')
except:
    pass  # Ignore if running locally (file is saved to the same directory)

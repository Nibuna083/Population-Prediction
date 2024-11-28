# model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

def train_model():
    # Load the dataset
    df = pd.read_csv(r"D:\New folder\world_population_analysis\data\world_population.csv")

    # Select features and target
    features = df[['Area (km²)', 'Density (per km²)', 'Growth Rate']]
    target = df['2022 Population']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    # Train the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Save the model
    joblib.dump(model, 'population_model.pkl')

if __name__ == "__main__":
    train_model()
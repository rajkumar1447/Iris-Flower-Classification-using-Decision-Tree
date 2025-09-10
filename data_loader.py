import pandas as pd
from sklearn.datasets import load_iris

def load_data():
    """Load Iris dataset into a pandas DataFrame."""
    data = load_iris()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['species'] = data.target
    return df, data

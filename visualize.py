import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

def visualize_tree(classifier, data):
    """Plot and display the Decision Tree."""
    plt.figure(figsize=(12, 8))
    plot_tree(
        classifier,
        feature_names=data.feature_names,
        class_names=data.target_names,
        filled=True
    )
    plt.title("Decision Tree for Iris Flower Classification")
    plt.show()

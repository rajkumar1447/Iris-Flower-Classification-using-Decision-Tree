from data_loader import load_data
from model import train_and_evaluate
from visualize import visualize_tree

def main():
    df, data = load_data()
    classifier, accuracy, conf_matrix, class_report = train_and_evaluate(df)

    print(f"\nAccuracy: {accuracy * 100:.2f}%")
    print("\nConfusion Matrix:")
    print(conf_matrix)
    print("\nClassification Report:")
    print(class_report)

    visualize_tree(classifier, data)

if __name__ == "__main__":
    main()

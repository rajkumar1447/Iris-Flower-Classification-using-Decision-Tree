from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

def train_and_evaluate(df):
    """Train Decision Tree model and evaluate performance."""
    X = df.drop('species', axis=1)
    y = df['species']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    classifier = DecisionTreeClassifier()
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)

    accuracy = metrics.accuracy_score(y_test, y_pred)
    conf_matrix = metrics.confusion_matrix(y_test, y_pred)
    class_report = metrics.classification_report(y_test, y_pred)

    return classifier, accuracy, conf_matrix, class_report

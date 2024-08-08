# Import necessary libraries
import numpy as np
import pandas as pd


# Function for dataset transformation
def dataset_transformation(dataframe):
    # Age categorization
    age_conditions = [
        (dataframe['Age'] >= 30) & (dataframe['Age'] <= 50),
        (dataframe['Age'] < 30),
        (dataframe['Age'] > 50)
    ]
    age_categories = ['1', '0', '2']
    dataframe['Age'] = np.select(age_conditions, age_categories, default='unknown')
    dataframe['Age'] = pd.to_numeric(dataframe['Age'], downcast='integer')

    # Sex conversion
    sex_mapping = {'F': '1', 'M': '0'}
    dataframe['Sex'] = dataframe['Sex'].map(sex_mapping).fillna('unknown')
    dataframe['Sex'] = pd.to_numeric(dataframe['Sex'], downcast='integer')

    # Blood Pressure categorization
    bp_mapping = {'HIGH': '2', 'NORMAL': '1', 'LOW': '0'}
    dataframe['BP'] = dataframe['BP'].map(bp_mapping).fillna('unknown')
    dataframe['BP'] = pd.to_numeric(dataframe['BP'], downcast='integer')

    # Cholesterol level adjustment
    cholesterol_mapping = {'HIGH': '1', 'NORMAL': '0'}
    dataframe['Cholesterol'] = dataframe['Cholesterol'].map(cholesterol_mapping).fillna('unknown')
    dataframe['Cholesterol'] = pd.to_numeric(dataframe['Cholesterol'], errors='coerce')

    # Drug encoding
    drug_mapping = {'drugA': '0', 'drugC': '2', 'drugX': '3', 'drugB': '1', 'drugY': '4'}
    dataframe['Drug'] = dataframe['Drug'].map(drug_mapping).fillna('unknown')
    dataframe['Drug'] = pd.to_numeric(dataframe['Drug'], errors='coerce')

    return dataframe

# Binary logistic regression model
class BinaryLogisticRegression:
    def __init__(self):
        self.weights = None

    def optimize(self, features, targets, iterations):
        features = np.hstack([np.ones((features.shape[0], 1)), features])  # Add bias column
        self.weights = np.zeros(features.shape[1])

        for _ in range(iterations):
            predictions = self.logistic_function(np.dot(features, self.weights))
            errors = predictions - targets
            gradient = np.dot(features.T, errors) / len(targets)
            self.weights -= 0.1 * gradient

    def logistic_function(self, z):
        return 1 / (1 + np.exp(-z))

    def predict(self, features):
        features = np.hstack([np.ones((features.shape[0], 1)), features])  # Add bias column
        probabilities = self.logistic_function(np.dot(features, self.weights))
        return (probabilities >= 0.5).astype(int)

# Multiclass logistic regression using one-vs-rest strategy
class MulticlassLogisticRegression:
    def __init__(self):
        self.classifiers = {}

    def train(self, features, targets, iterations):
        unique_targets = np.unique(targets)
        for target in unique_targets:
            binary_targets = (targets == target).astype(int)
            classifier = BinaryLogisticRegression()
            classifier.optimize(features, binary_targets, iterations)
            self.classifiers[target] = classifier

    def predict(self, features):
        predictions = np.zeros((features.shape[0], len(self.classifiers)))
        for target, classifier in self.classifiers.items():
            predictions[:, target] = classifier.predict(features)
        return np.argmax(predictions, axis=1)
    


def custom_accuracy(y_true, y_pred):
    correct = 0
    total = len(y_true)
    for true, pred in zip(y_true, y_pred):
        if true == pred:
            correct += 1
    return correct / total

# Load the dataset
raw_data = pd.read_csv('drug200.csv')
processed_data = dataset_transformation(raw_data)

# Separate the features and the target variable
X = processed_data.drop(columns=['Drug'])  # Convert to numpy array
y = processed_data['Drug']  # Convert to numpy array


test_size = 0.18
random_state = 0

# Calculate the index to split the data
split_index = int(len(processed_data) * (1 - test_size))
train_attributes, test_attributes = pd.DataFrame(), pd.DataFrame()
train_labels, test_labels = pd.DataFrame(), pd.DataFrame()

# Iterate over the indices
test_size = 0.1
random_state = 42

# Calculate the index to split the data
split_index = int(len(X) * (1 - test_size))

# Split the data into train and test sets
X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]
y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]


# Converting them into np.arrays
X_train=X_train.values
X_test=X_test.values
y_test=y_test.values 
y_train=y_train.values



# Initialize and train the multiclass logistic regression model
model = MulticlassLogisticRegression()
model.train(X_train, y_train, 200)  # 200 iterations

# Evaluate the model
y_pred = model.predict(X_test)

#Calling the custom acc_function
accuracy = custom_accuracy(y_test, y_pred)
print(f'Accuracy on the test set: {accuracy:.2f}')

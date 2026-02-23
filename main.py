# Step 1: libraries import
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Step 2: dataset load
iris = load_iris()
X = iris.data
y = iris.target

# Step 3: training and testing split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Step 4: model create and train
model = RandomForestClassifier()
model.fit(X_train, y_train)
joblib.dump(model, "iris_model.pkl")
print("Model saved successfully!")
# Load saved model
model = joblib.load("iris_model.pkl")
print("Model loaded successfully!")

# Step 5: prediction
predictions = model.predict(X_test)

# Step 6: accuracy check
accuracy = accuracy_score(y_test, predictions)
print("Model Accuracy: {:.2f}%".format(accuracy * 100))

print("\nEnter flower features:")

sepal_length = float(input("Sepal length: "))
sepal_width = float(input("Sepal width: "))
petal_length = float(input("Petal length: "))
petal_width = float(input("Petal width: "))

sample = [[sepal_length, sepal_width, petal_length, petal_width]]

result = model.predict(sample)
print("Predicted flower type:", iris.target_names[result][0])
# Step 8: data visualization
import matplotlib.pyplot as plt

# sepal length vs sepal width plot
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.title("Iris Flower Visualization")

plt.savefig("iris_graph.png")
print("Graph saved as iris_graph.png")
plt.show()

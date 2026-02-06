import torch
import torch.nn as nn
import torch.optim as optim
import csv
from sa_nn import export_sa_nn

# Load Iris dataset
def load_iris_data():
    # Define the classic iris dataset
    iris_data = [
        [5.1, 3.5, 1.4, 0.2, 0],  # setosa
        [4.9, 3.0, 1.4, 0.2, 0],
        [4.7, 3.2, 1.3, 0.2, 0],
        [4.6, 3.1, 1.5, 0.2, 0],
        [5.0, 3.6, 1.4, 0.2, 0],
        [5.4, 3.9, 1.7, 0.4, 0],
        [4.6, 3.4, 1.4, 0.3, 0],
        [5.0, 3.4, 1.5, 0.2, 0],
        [4.4, 2.9, 1.4, 0.2, 0],
        [4.9, 3.1, 1.5, 0.1, 0],
        [5.4, 3.7, 1.5, 0.2, 0],
        [4.8, 3.4, 1.6, 0.2, 0],
        [4.8, 3.0, 1.4, 0.1, 0],
        [4.3, 3.0, 1.1, 0.1, 0],
        [5.8, 4.0, 1.2, 0.2, 0],
        [5.7, 4.4, 1.5, 0.4, 0],
        [5.4, 3.9, 1.3, 0.4, 0],
        [5.1, 3.5, 1.4, 0.3, 0],
        [5.7, 3.8, 1.7, 0.3, 0],
        [5.1, 3.8, 1.5, 0.3, 0],
        [5.4, 3.4, 1.7, 0.2, 0],
        [5.1, 3.7, 1.5, 0.4, 0],
        [4.6, 3.6, 1.0, 0.2, 0],
        [5.1, 3.3, 1.7, 0.5, 0],
        [4.8, 3.4, 1.9, 0.2, 0],
        [5.0, 3.0, 1.6, 0.2, 0],
        [5.0, 3.4, 1.6, 0.4, 0],
        [5.2, 3.5, 1.5, 0.2, 0],
        [5.2, 3.4, 1.4, 0.2, 0],
        [4.7, 3.2, 1.6, 0.2, 0],
        [4.8, 3.1, 1.6, 0.2, 0],
        [5.4, 3.4, 1.5, 0.4, 0],
        [5.2, 4.1, 1.5, 0.1, 0],
        [5.5, 4.2, 1.4, 0.2, 0],
        [4.9, 3.1, 1.5, 0.1, 0],
        [5.0, 3.2, 1.2, 0.2, 0],
        [5.5, 3.5, 1.3, 0.2, 0],
        [4.9, 3.1, 1.5, 0.1, 0],
        [4.4, 3.0, 1.3, 0.2, 0],
        [5.1, 3.4, 1.5, 0.2, 0],
        [5.0, 3.5, 1.3, 0.3, 0],
        [4.5, 2.3, 1.3, 0.3, 0],
        [4.4, 3.2, 1.3, 0.2, 0],
        [5.0, 3.5, 1.6, 0.6, 0],
        [5.1, 3.8, 1.9, 0.4, 0],
        [4.8, 3.0, 1.4, 0.3, 0],
        [5.1, 3.8, 1.6, 0.2, 0],
        [4.6, 3.2, 1.4, 0.2, 0],
        [5.3, 3.7, 1.5, 0.2, 0],
        [5.0, 3.3, 1.4, 0.2, 0],
        [7.0, 3.2, 4.7, 1.4, 1],  # versicolor
        [6.4, 3.2, 4.5, 1.5, 1],
        [6.9, 3.1, 4.9, 1.5, 1],
        [5.5, 2.3, 4.0, 1.3, 1],
        [6.5, 2.8, 4.6, 1.5, 1],
        [5.7, 2.8, 4.5, 1.3, 1],
        [6.3, 3.3, 4.7, 1.6, 1],
        [4.9, 2.4, 3.3, 1.0, 1],
        [6.6, 2.9, 4.6, 1.3, 1],
        [5.2, 2.7, 3.9, 1.4, 1],
        [5.0, 2.0, 3.5, 1.0, 1],
        [5.9, 3.0, 4.2, 1.5, 1],
        [6.0, 2.2, 4.0, 1.0, 1],
        [6.1, 2.9, 4.7, 1.4, 1],
        [5.6, 2.9, 3.6, 1.3, 1],
        [6.7, 3.1, 4.4, 1.4, 1],
        [5.6, 3.0, 4.5, 1.5, 1],
        [5.8, 2.7, 4.1, 1.0, 1],
        [6.2, 2.2, 4.5, 1.5, 1],
        [5.6, 2.5, 3.9, 1.1, 1],
        [5.9, 3.2, 4.8, 1.8, 1],
        [6.1, 2.8, 4.0, 1.3, 1],
        [6.3, 2.5, 4.9, 1.5, 1],
        [6.1, 2.8, 4.7, 1.2, 1],
        [6.4, 2.9, 4.3, 1.3, 1],
        [6.6, 3.0, 4.4, 1.4, 1],
        [6.8, 2.8, 4.8, 1.4, 1],
        [6.7, 3.0, 5.0, 1.7, 1],
        [6.0, 2.9, 4.5, 1.5, 1],
        [5.7, 2.6, 3.5, 1.0, 1],
        [5.5, 2.4, 3.8, 1.1, 1],
        [5.5, 2.4, 3.7, 1.0, 1],
        [5.8, 2.7, 3.9, 1.2, 1],
        [6.0, 2.7, 5.1, 1.6, 1],
        [5.4, 3.0, 4.5, 1.5, 1],
        [6.0, 3.4, 4.5, 1.6, 1],
        [6.7, 3.1, 4.7, 1.5, 1],
        [6.3, 2.3, 4.4, 1.3, 1],
        [5.6, 3.0, 4.1, 1.3, 1],
        [5.5, 2.5, 4.0, 1.3, 1],
        [5.5, 2.6, 4.4, 1.2, 1],
        [6.1, 3.0, 4.6, 1.4, 1],
        [5.8, 2.6, 4.0, 1.2, 1],
        [5.0, 2.3, 3.3, 1.0, 1],
        [5.6, 2.7, 4.2, 1.3, 1],
        [5.7, 3.0, 4.2, 1.2, 1],
        [5.7, 2.9, 4.2, 1.3, 1],
        [6.2, 2.9, 4.3, 1.3, 1],
        [5.1, 2.5, 3.0, 1.1, 1],
        [5.7, 2.8, 4.1, 1.3, 1],
        [6.3, 3.3, 6.0, 2.5, 2],  # virginica
        [5.8, 2.7, 5.1, 1.9, 2],
        [7.1, 3.0, 5.9, 2.1, 2],
        [6.3, 2.9, 5.6, 1.8, 2],
        [6.5, 3.0, 5.8, 2.2, 2],
        [7.6, 3.0, 6.6, 2.1, 2],
        [4.9, 2.5, 4.5, 1.7, 2],
        [7.3, 2.9, 6.3, 1.8, 2],
        [6.7, 2.5, 5.8, 1.8, 2],
        [7.2, 3.6, 6.1, 2.5, 2],
        [6.5, 3.2, 5.1, 2.0, 2],
        [6.4, 2.7, 5.3, 1.9, 2],
        [6.8, 3.0, 5.5, 2.1, 2],
        [5.7, 2.5, 5.0, 2.0, 2],
        [5.8, 2.8, 5.1, 2.4, 2],
        [6.4, 3.2, 5.3, 2.3, 2],
        [6.5, 3.0, 5.5, 1.8, 2],
        [7.7, 3.8, 6.7, 2.2, 2],
        [7.7, 2.6, 6.9, 2.3, 2],
        [6.0, 2.2, 5.0, 1.5, 2],
        [6.9, 3.2, 5.7, 2.3, 2],
        [5.6, 2.8, 4.9, 2.0, 2],
        [7.7, 2.8, 6.7, 2.0, 2],
        [6.3, 2.7, 4.9, 1.8, 2],
        [6.7, 3.3, 5.7, 2.1, 2],
        [7.2, 3.2, 6.0, 1.8, 2],
        [6.2, 2.8, 4.8, 1.8, 2],
        [6.1, 3.0, 4.9, 1.8, 2],
        [6.4, 2.8, 5.6, 2.1, 2],
        [7.2, 3.0, 5.8, 1.6, 2],
        [7.4, 2.8, 6.1, 1.9, 2],
        [7.9, 3.8, 6.4, 2.0, 2],
        [6.4, 2.8, 5.6, 2.2, 2],
        [6.3, 2.8, 5.1, 1.5, 2],
        [6.1, 2.6, 5.6, 1.4, 2],
        [7.7, 3.0, 6.1, 2.3, 2],
        [6.3, 3.4, 5.6, 2.4, 2],
        [6.4, 3.1, 5.5, 1.8, 2],
        [6.0, 3.0, 4.8, 1.8, 2],
        [6.9, 3.1, 5.4, 2.1, 2],
        [6.7, 3.1, 5.6, 2.4, 2],
        [6.9, 3.1, 5.1, 2.3, 2],
        [5.8, 2.7, 5.1, 1.9, 2],
        [6.8, 3.2, 5.9, 2.3, 2],
        [6.7, 3.3, 5.7, 2.5, 2],
        [6.7, 3.0, 5.2, 2.3, 2],
        [6.3, 2.5, 5.0, 1.9, 2],
        [6.5, 3.0, 5.2, 2.0, 2],
        [6.2, 3.4, 5.4, 2.3, 2],
        [5.9, 3.0, 5.1, 1.8, 2]
    ]
    
    X = []
    y = []
    for row in iris_data:
        X.append(row[:-1])  # Features: sepal length, sepal width, petal length, petal width
        y.append(row[-1])   # Labels: 0=setosa, 1=versicolor, 2=virginica
    
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

# Define a 4-layer neural network for Iris classification
class IrisNet4Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 8)      # Input: 4 features -> Hidden: 8 neurons
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(8, 12)     # Hidden: 8 -> Hidden: 12 neurons
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(12, 8)     # Hidden: 12 -> Hidden: 8 neurons
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(8, 3)      # Hidden: 8 -> Output: 3 classes
    
    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.relu3(self.fc3(x))
        x = self.fc4(x)
        return x

# Load data
X, y = load_iris_data()
print(f"Iris dataset loaded: {X.shape[0]} samples, {X.shape[1]} features, {len(torch.unique(y))} classes")

# Create and train the model
model = IrisNet4Layer()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

print("Training 4-layer Iris classification model...")

# Training loop for 25 epochs
for epoch in range(25):
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 5 == 0:
        print(f'Epoch [{epoch+1}/25], Loss: {loss.item():.4f}')

# Test the trained model
model.eval()
with torch.no_grad():
    predictions = model(X)
    _, predicted = torch.max(predictions.data, 1)
    accuracy = (predicted == y).float().mean()
    print(f'Training Accuracy: {accuracy:.4f}')

# Export the model using SA-NN with PROGMEM support
print("\nExporting 4-layer Iris model to SA-NN format with PROGMEM...")
export_sa_nn(model, filename="iris_model.h", use_progmem=True)
print("4-layer Iris model with PROGMEM exported successfully!")

# Test with some sample inputs
test_inputs = [
    [5.1, 3.5, 1.4, 0.2],  # Expected: setosa (class 0)
    [7.0, 3.2, 4.7, 1.4],  # Expected: versicolor (class 1) 
    [6.3, 3.3, 6.0, 2.5]   # Expected: virginica (class 2)
]

class_names = ["setosa", "versicolor", "virginica"]

print("\nTesting trained model on sample inputs:")
model.eval()
with torch.no_grad():
    for i, test_input in enumerate(test_inputs):
        x_test = torch.tensor([test_input], dtype=torch.float32)
        output = model(x_test)
        probabilities = torch.softmax(output, dim=1)
        prediction = torch.argmax(probabilities, dim=1)
        
        predicted_class = class_names[prediction.item()]
        confidence = probabilities[0][prediction.item()].item()
        
        print(f"Sample {i+1}: {test_input} -> {predicted_class} (confidence: {confidence:.3f})")
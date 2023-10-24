import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

# Read the dataset
data = pd.read_csv('water.csv')
data = shuffle(data)  # Shuffling the dataset

# Extract the input features and labels
X = data.drop('Potability', axis=1).values
y = data['Potability'].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert the data into PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Define the model architecture and hyperparameters
class WaterClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(WaterClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

input_size = 9  # Adjust according to the number of input features
hidden_size = 64  # You can adjust this based on experimentation
output_size = 2  # Assuming binary classification for drinkable vs. undrinkable water

# Initialize the model
model = WaterClassifier(input_size, hidden_size, output_size)

# Define the loss function and the optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0000001)  # You can adjust the learning rate as needed

# Training loop
num_epochs = 1000  # Define the appropriate number of epochs
for epoch in range(num_epochs):
    inputs = X_train_tensor
    labels = y_train_tensor

    # Forward pass
    outputs = model(inputs)
    loss = criterion(outputs, labels)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Print the loss at every 100 epochs or as needed
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# Evaluate the model
with torch.no_grad():
    outputs = model(X_test_tensor)
    _, predicted = torch.max(outputs, 1)
    total = y_test_tensor.size(0)
    correct = (predicted == y_test_tensor).sum().item()
    accuracy = correct / total
    print(f'Accuracy on the test set: {accuracy:.2f}')

# Save the trained model
torch.save(model.state_dict(), 'water_classifier_model.model')

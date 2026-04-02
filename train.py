import torch
import torch.nn as nn
import torch.optim as optim

from data.load_data import get_data
from models.cnn_model import CNN


train_loader, test_loader = get_data()

model = CNN()

loss_function = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=0.001)


epochs = 5

for epoch in range(epochs):

    for images, labels in train_loader:

        # Clear gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)

        # Compute loss
        loss = loss_function(outputs, labels)

        # Backpropagation
        loss.backward()

        # Update weights
        optimizer.step()

    print("Epoch:", epoch+1, "Loss:", loss.item())


torch.save(model.state_dict(), "digit_cnn_model.pth")

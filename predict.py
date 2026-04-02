import torch

from models.cnn_model import CNN
from data.load_data import get_data

model = CNN()
model.load_state_dict(torch.load("digit_cnn_model.pth"))

model.eval()

_, test_loader = get_data()

correct = 0
total = 0

with torch.no_grad():

    for images, labels in test_loader:

        outputs = model(images)

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)

        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total

print("Model Accuracy:", accuracy, "%")
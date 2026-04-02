import torch
import matplotlib.pyplot as plt

from models.cnn_model import CNN
from data.load_data import get_data

model = CNN()
model.load_state_dict(torch.load("digit_cnn_model.pth"))

model.eval()

_, test_loader = get_data()

images, labels = next(iter(test_loader))

outputs = model(images)

_, predicted = torch.max(outputs, 1)

for i in range(6):

    plt.subplot(2,3,i+1)
    plt.imshow(images[i].squeeze(), cmap="gray")
    plt.title(f"P:{predicted[i].item()} A:{labels[i].item()}")
    plt.axis("off")

plt.show()
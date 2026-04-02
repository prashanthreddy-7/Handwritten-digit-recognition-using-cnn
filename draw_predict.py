import tkinter as tk
import torch
import numpy as np
from PIL import Image
from models.cnn_model import CNN
from torchvision import transforms
from PIL import ImageGrab


# load trained model
model = CNN()
model.load_state_dict(torch.load("digit_cnn_model.pth"))
model.eval()


# transformation for MNIST format
transform = transforms.Compose([
    transforms.Resize((28,28)),
    transforms.Grayscale(),
    transforms.ToTensor()
])


# predict function
def predict_digit():

    # get canvas position on screen
    x = root.winfo_rootx() + canvas.winfo_x()
    y = root.winfo_rooty() + canvas.winfo_y()

    x1 = x + canvas.winfo_width()
    y1 = y + canvas.winfo_height()

    # capture canvas area
    img = ImageGrab.grab().crop((x, y, x1, y1))
    img = img.convert("L")   # convert to grayscale
    img = Image.eval(img, lambda x: 255 - x)  # invert colors
    img = transform(img)

    img = img.unsqueeze(0)

    output = model(img)

    _, predicted = torch.max(output,1)

    result_label.config(text=f"Prediction: {predicted.item()}")


# draw on canvas
def draw(event):

    x1 = event.x - 8
    y1 = event.y - 8
    x2 = event.x + 8
    y2 = event.y + 8

    canvas.create_oval(x1,y1,x2,y2,fill="black")


# clear canvas
def clear():

    canvas.delete("all")
    result_label.config(text="Draw a digit")


# GUI
root = tk.Tk()
root.title("CNN Digit Recognizer")

canvas = tk.Canvas(root,width=200,height=200,bg="white")
canvas.pack()

canvas.bind("<B1-Motion>",draw)

predict_button = tk.Button(root,text="Predict",command=predict_digit)
predict_button.pack()

clear_button = tk.Button(root,text="Clear",command=clear)
clear_button.pack()

result_label = tk.Label(root,text="Draw a digit",font=("Helvetica",16))
result_label.pack()

root.mainloop()
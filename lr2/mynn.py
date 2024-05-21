import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F  # Добавим импорт функции relu
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import Canvas, Button
from PIL import Image, ImageDraw
import io

# Определение преобразований для изображений
transform = transforms.Compose([
    transforms.Grayscale(),  # Преобразование изображений в черно-белые
    transforms.Resize((32, 32)),  # Изменение размера до 32x32
    transforms.ToTensor()  # Преобразование изображений в тензоры
])

# Загрузка датасета
dataset = ImageFolder(root= r'C:\Users\bladp\Desktop\ai\lr2\dataset', transform=transform)

# Разделение датасета на тренировочный и тестовый
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

# Создание DataLoader для тренировочного и тестового датасетов
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
classes = train_loader.dataset



class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)  # 10 классов символов хираганы

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = CNN()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10

# Создаем списки для сохранения Loss и Accuracy
train_loss_values = []
test_loss_values = []
train_accuracy_values = []
test_accuracy_values = []
if __name__ == "__main__":
    # Цикл обучения
    for epoch in range(num_epochs):
        running_train_loss = 0.0
        running_test_loss = 0.0
        correct_train = 0
        total_train = 0
        correct_test = 0
        total_test = 0
        
        # Тренировочная фаза
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
        
        train_loss_values.append(running_train_loss / len(train_loader))
        train_accuracy_values.append(100 * correct_train / total_train)
        
        # Тестовая фаза
        with torch.no_grad():
            for data in test_loader:
                inputs, labels = data
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_test += labels.size(0)
                correct_test += (predicted == labels).sum().item()
            
        test_loss_values.append(running_test_loss / len(test_loader))
        test_accuracy_values.append(100 * correct_test / total_test)

    torch.save(model.state_dict(), 'model.pth')

    # Построение графиков после завершения обучения
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_loss_values, label='Train Loss')
    plt.plot(test_loss_values, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train and Test Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracy_values, label='Train Accuracy')
    plt.plot(test_accuracy_values, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Train and Test Accuracy')
    plt.legend()

    plt.show()


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tkinter as tk
from PIL import Image, ImageOps, ImageTk
import io
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        self.weights_input_hidden = np.random.randn(self.input_size, self.hidden_size)
        self.weights_hidden_output = np.random.randn(self.hidden_size, self.output_size)
        
        self.bias_input_hidden = np.zeros((1, self.hidden_size))
        self.bias_hidden_output = np.zeros((1, self.output_size))
        
        # Метрики
        self.accuracy = 0
        self.total_samples = 0
        self.correct_predictions = 0
        
        # Добавляем списки для хранения значений loss и accuracy
        self.loss_history = []
        self.accuracy_history = []
        
    def calculate_accuracy(self, X, y):
        # Прямое прохождение через сеть
        output = self.forward(X)
        
        # Получаем предсказанные классы
        predicted_classes = np.argmax(output, axis=1)
        
        # Считаем количество правильных предсказаний
        correct = (predicted_classes == y).sum()
        
        # Обновляем переменные для хранения метрик
        self.total_samples += len(y)
        self.correct_predictions += correct
        self.accuracy = self.correct_predictions / self.total_samples
        self.accuracy_history.append(self.accuracy)
    
    def forward(self, X):
        self.hidden = sigmoid(np.dot(X, self.weights_input_hidden) + self.bias_input_hidden)
        self.output = sigmoid(np.dot(self.hidden, self.weights_hidden_output) + self.bias_hidden_output)
        return self.output
    
    def backward(self, X, y, learning_rate):
        error = y - self.output
        d_output = error * sigmoid_derivative(self.output)
        
        error_hidden = np.dot(d_output, self.weights_hidden_output.T)
        d_hidden = error_hidden * sigmoid_derivative(self.hidden)
        
        self.weights_hidden_output += np.dot(self.hidden.T, d_output) * learning_rate
        self.bias_hidden_output += np.sum(d_output, axis=0, keepdims=True) * learning_rate
        
        self.weights_input_hidden += np.dot(X.T, d_hidden) * learning_rate
        self.bias_input_hidden += np.sum(d_hidden, axis=0, keepdims=True) * learning_rate
        
    def train(self, X, y, epochs, learning_rate):
        for epoch in range(epochs):
            self.forward(X)
            self.backward(X, y, learning_rate)
            if epoch % 100 == 0:
                loss = np.mean(np.square(y - self.forward(X)))
                self.calculate_accuracy(X, np.argmax(y, axis=1))  # добавляем вычисление точности
                self.loss_history.append(loss)  # добавляем значение потерь в историю
                print(f"Epoch {epoch}, Loss: {loss}, Accuracy: {self.accuracy}")
        
        # Вывод графиков
        self.plot_loss_accuracy()
    
    def evaluate(self, X, y):
        predictions = self.forward(X)
        loss = np.mean(np.square(y - predictions))
        
        # Расчет accuracy
        correct_predictions = np.sum(np.argmax(predictions, axis=1) == np.argmax(y, axis=1))
        total_predictions = len(y)
        accuracy = correct_predictions / total_predictions
        
        print(f"Loss: {loss}, Accuracy: {accuracy}")
        
        return loss, accuracy
    
    def plot_loss_accuracy(self):
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(range(len(self.loss_history)), self.loss_history)
        plt.title('Loss')
        plt.xlabel('Epoch*100')
        plt.ylabel('Loss')
        
        plt.subplot(1, 2, 2)
        plt.plot(range(len(self.accuracy_history)), self.accuracy_history)
        plt.title('Accuracy')
        plt.xlabel('Epoch*100')
        plt.ylabel('Accuracy')
        
        plt.show()


# Загрузка данных из аннотации
# annotation_data = pd.read_csv(r'C:\Users\bladp\Desktop\ai\ai_lr2\annotation_v2_comma.csv')
annotation_data = pd.read_csv(r'C:\Users\bladp\Desktop\ai\ai_lr2\annotation.csv', names=['Pixel Vector', 'Class Label'], sep=';')
print(annotation_data.columns)
# Получение векторов пикселей и меток классов
pixel_vectors = annotation_data['Pixel Vector'].values
labels = annotation_data['Class Label'].values


# Преобразование меток классов в уникальные целочисленные идентификаторы
label_to_id = {label: idx for idx, label in enumerate(np.unique(labels))}
id_to_label = {idx: label for label, idx in label_to_id.items()}
label_ids = np.array([label_to_id[label] for label in labels])

# Преобразование векторов пикселей в массивы Numpy
pixel_matrices = np.array([np.array([int(pixel) for pixel in vector.split(',')]) for vector in pixel_vectors])

# Преобразование меток классов в формат One-Hot Encoding
num_classes = len(np.unique(labels))
one_hot_labels = np.eye(num_classes)[label_ids]

# Разделение данных на обучающий, валидационный и тестовый наборы
X_train_val, X_test, y_train_val, y_test = train_test_split(pixel_matrices, one_hot_labels, test_size=0.2, random_state=1234)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=1234)  # 0.25 * 0.8 = 0.2


# Сетка
input_size = 32 * 32  # размер вектора пикселей
hidden_size = 16  # количество нейронов в скрытом слое
output_size = num_classes  # количество классов 

nn = NeuralNetwork(input_size, hidden_size, output_size)

# Обучение нейронной сети на обучающем наборе
epochs = 2000
learning_rate = 0.02

nn.train(X_train, y_train, epochs, learning_rate)

# Оценка производительности на валидационном наборе
print("Оценка производительности на валидационном наборе:")
val_loss, val_accuracy = nn.evaluate(X_val, y_val)

# Оценка производительности на тестовом наборе
print("Оценка производительности на тестовом наборе:")
test_loss, test_accuracy = nn.evaluate(X_test, y_test)





def recognize():
    predictionText.delete('1.0', 'end')
    # Получаем изображение с холста и сохраняем его в формате PNG
    canvas_image = canvas.postscript(colormode='color')
    img = Image.open(io.BytesIO(canvas_image.encode('utf-8')))
    
    # Преобразуем изображение в оттенки серого
    img = ImageOps.grayscale(img)
    
    # Инвертируем цвета, чтобы символ был белым на черном фоне
    img = ImageOps.invert(img)
    
    # Находим границы нарисованного символа с отступом
    bbox = img.getbbox()
    
    # Рассчитываем новые размеры изображения с сохранением соотношения сторон
    width, height = bbox[2] - bbox[0], bbox[3] - bbox[1]
    max_size = max(width, height)
    new_width = int(32 * width / max_size)
    new_height = int(32 * height / max_size)
    offset_x = (32 - new_width) // 2
    offset_y = (32 - new_height) // 2
    
    # Добавляем отступы к обрезанному изображению
    bbox_with_padding = (bbox[0] - 7, bbox[1] - 7, bbox[2] + 7, bbox[3] + 7)
    
    # Обрезаем и изменяем размер изображения с учетом отступов
    img_cropped_resized = img.crop(bbox_with_padding).resize((new_width, new_height))
    
    # Создаем новое изображение 32x32 и вставляем обрезанное изображение в центр
    img_final = Image.new('L', (32, 32), color=255)  # создаем белое изображение 32x32
    img_final.paste(ImageOps.invert(img_cropped_resized), (offset_x, offset_y))
    
    # Сохраняем обрезанное и измененное изображение на диск
    img_final.save("cropped_symbol.png")
    
    # Преобразуем изображение в вектор пикселей
    pixel_vector = np.array(img_final).flatten()
    
    # Нормализуем значения пикселей
    for i in range(len(pixel_vector)):
         if pixel_vector[i] > 127:
             pixel_vector[i] = 0
         else:
             pixel_vector[i] = 1
    
    # Пропускаем вектор пикселей через нейронную сеть
    output = nn.forward(pixel_vector)
    
    # Получаем предсказание числовного класса символа
    predicted_class = np.argmax(output)
    
    # Получаем метку класса по числовому значению из словаря
    predicted_label = id_to_label[predicted_class]
    
    # Отображаем предсказание 
    print(f"Распознанный символ: {predicted_label}")
    predictionText.insert("1.0", predicted_label)
    
# Подготовка пользовательского интерфейса
root = tk.Tk()
root.title("Symbol Recognition")

canvas = tk.Canvas(root, width=400, height=400, bg='white')
canvas.pack()

def draw(event):
    x, y = event.x, event.y
    r = brush_size.get()  # радиус кисти, получаем значение с ползунка
    canvas.create_oval(x-r, y-r, x+r, y+r, fill='black')
    
# Функция для очистки холста
def clear_canvas():
    canvas.delete("all")

canvas.bind('<B1-Motion>', draw)

recognize_button = tk.Button(root, text="Recognize", command=recognize)
recognize_button.pack(side = 'left')


# Кнопка для очистки холста
clear_button = tk.Button(root, text="Clear", command=clear_canvas)
clear_button.pack(side = 'right')

# Ползунок для изменения ширины кисти
brush_size = tk.Scale(root, from_=1, to=10, orient="horizontal", label="Width")
brush_size.set(5)
brush_size.pack()

predictionText = tk.Text(root, height=1, width=10, )
predictionText.pack()

lossText = tk.Text(root, height=1, width=40, )
lossText.pack(side='left')
lossText.insert("1.0", f"Test loss: {np.around(test_loss, 3)}, Test accuracy: {np.around(test_accuracy, 3)}")


root.mainloop()


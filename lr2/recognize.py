import io
import tkinter as tk
from tkinter import Canvas, Button
from PIL import Image, ImageDraw
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageOps, ImageTk
from mynn import CNN

# Преобразование для изображения
transform = transforms.Compose([
    transforms.Grayscale(),  # Преобразование изображений в черно-белые
    transforms.Resize((32, 32)),  # Изменение размера до 32x32
    transforms.ToTensor()  # Преобразование изображений в тензоры
])

class_to_symbol = {
    0: 'chi',
    1: 'hi',
    2: 'ke',
    3: 'ku',
    4: 'n',
    5: 'no',
    6: 'ro',
    7: 'te',
    8: 'to',
    9: 'yo',
    # Далее продолжайте для всех ваших классов
}


# Загрузка сохраненной модели
model = CNN()  # Замените на ваш класс модели
model.load_state_dict(torch.load('model.pth'))
model.eval()  # Установка модели в режим оценки



def recognize():
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
    img_final.save("cropped_symbol.bmp")
    
    
    
    # Пропускаем вектор пикселей через нейронную сеть
    # output = nn.forward(pixel_vector)
    # Преобразование изображения и определение символа с помощью нейронной сети
    img_final = transform(img_final)
    img_final = img_final.unsqueeze(0)  # Добавляем размерность батча
    outputs = model(img_final)
    _, predicted = torch.max(outputs, 1)
    predicted_symbol = class_to_symbol[predicted.item()]
    symbol_label.config(text=f"Predicted symbol: {predicted_symbol}")
    
    

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

# Метка для отображения распознанного символа
symbol_label = tk.Label(root, text="")
symbol_label.pack()

# Кнопка для очистки холста
clear_button = tk.Button(root, text="Clear", command=clear_canvas)
clear_button.pack(side = 'right')
 
# Ползунок для изменения ширины кисти
brush_size = tk.Scale(root, from_=1, to=10, orient="horizontal", label="Width")
brush_size.set(5)
brush_size.pack()


root.mainloop()

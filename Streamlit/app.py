import streamlit as st
import torch
import torch.nn.functional as f
from PIL import Image
from torchvision.transforms import v2
from torchvision.transforms.functional import pil_to_tensor

d_inv = {0: 'ромашка', 1: 'одуванчик', 2: 'роза', 3: 'подсолнух', 4: 'тюльпан'}
model = torch.load('model.pth')
transforms = v2.Compose([
    v2.Resize(size=(227, 227), antialias=True),  # Or Resize(antialias=True)
    v2.ToDtype(torch.float32, scale=True),  # Normalize expects float input
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
st.title('Нейронная сеть для классификации цветов')
st.header('Загрузите изображение цветка:')
file = st.file_uploader("Выберите изображение")
image = None
if st.button("Определить", type="primary"):
    try:
        image = Image.open(file)
    except IOError:
        raise Exception('Данный файл не является изображением')
    st.image(file)
    im = pil_to_tensor(image)
    im = transforms(im)
    prediction = model(im.unsqueeze(0))
    _, predictions = torch.max(prediction.data, 1)
    prob = round(float((f.softmax(prediction, dim=1).max() * 100)))
    st.write(f'Я думаю, что на изображении находится {d_inv[predictions.numpy()[0]]} с вероятностью в {prob}%')

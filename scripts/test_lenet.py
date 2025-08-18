from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np

path_1 = './images/Prova/cartello13.jpg'
path_2 = './images/Prova/cartello17.jpg'
path_3 = './images/Prova/cartello19.jpg'
path_4 = './images/Prova/cartello25.jpg'
path_5 = './images/Prova/cartello39.jpg'
path_6 = './images/Prova/segnale25.jpeg'
path_7 = './images/Prova/segnale39.jpg'

img_1 = image.load_img(path_1, target_size=(32, 32))
img_2 = image.load_img(path_2, target_size=(32, 32))
img_3 = image.load_img(path_3, target_size=(32, 32))
img_4 = image.load_img(path_4, target_size=(32, 32))
img_5 = image.load_img(path_5, target_size=(32, 32))
img_6 = image.load_img(path_6, target_size=(32, 32))
img_7 = image.load_img(path_7, target_size=(32, 32))

arr_1 = image.img_to_array(img_1)
arr_2 = image.img_to_array(img_2)
arr_3 = image.img_to_array(img_3)
arr_4 = image.img_to_array(img_4)
arr_5 = image.img_to_array(img_5)
arr_6 = image.img_to_array(img_6)
arr_7 = image.img_to_array(img_7)

arr_1 /= 255.0
arr_2 /= 255.0
arr_3 /= 255.0
arr_4 /= 255.0
arr_5 /= 255.0
arr_6 /= 255.0
arr_7 /= 255.0

arr_1 = np.expand_dims(arr_1, axis=0)
arr_2 = np.expand_dims(arr_2, axis=0)
arr_3 = np.expand_dims(arr_3, axis=0)
arr_4 = np.expand_dims(arr_4, axis=0)
arr_5 = np.expand_dims(arr_5, axis=0)
arr_6 = np.expand_dims(arr_6, axis=0)
arr_7 = np.expand_dims(arr_7, axis=0)

# LeNet 5 Pro
model = load_model('./models/lenetpro.h5')
# LeNet 5 Plus
#model = load_model('./models/lenetplus.h5')
# LeNet 5
#model = load_model('./models/lenet.h5')

pred1 = model.predict(arr_1)
pred2 = model.predict(arr_2)
pred3 = model.predict(arr_3)
pred4 = model.predict(arr_4)
pred5 = model.predict(arr_5)
pred6 = model.predict(arr_6)
pred7 = model.predict(arr_7)

p1 = np.argmax(pred1)
p2 = np.argmax(pred2)
p3 = np.argmax(pred3)
p4 = np.argmax(pred4)
p5 = np.argmax(pred5)
p6 = np.argmax(pred6)
p7 = np.argmax(pred7)

print(f'Cartello 13: Classe predetta: {p1}')
print(f'Cartello 17: Classe predetta: {p2}')
print(f'Cartello 19: Classe predetta: {p3}')
print(f'Cartello 25: Classe predetta: {p4}')
print(f'Cartello 39: Classe predetta: {p5}')
print(f'Cartello 25 nuovo: Classe predetta: {p6}')
print(f'Cartello 39 nuovo: Classe predetta: {p7}')
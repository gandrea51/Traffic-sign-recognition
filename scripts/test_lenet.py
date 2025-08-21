from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np

# My photos
number3 = './images/Prova/3_personal.jpg'
number9 = './images/Prova/9_personal.jpg'
number39 = './images/Prova/39_personal.jpg'
number17 = './images/Prova/17_personal.jpg'
number31 = './images/Prova/31_personal.jpg'

# Other photos
number13 = './images/Prova/13_int.jpg'
number19 = './images/Prova/19_int.jpg'
number25 = './images/Prova/25_int.jpg'
number25bis = './images/Prova/25_temp.jpeg'
number39bis = './images/Prova/39_temp.jpg'


number3 = image.load_img(number3, target_size=(32, 32))
number9 = image.load_img(number9, target_size=(32, 32))
number39 = image.load_img(number39, target_size=(32, 32))
number17 = image.load_img(number17, target_size=(32, 32))
number31 = image.load_img(number31, target_size=(32, 32))

number13 = image.load_img(number13, target_size=(32, 32))
number19 = image.load_img(number19, target_size=(32, 32))
number25 = image.load_img(number25, target_size=(32, 32))
number25bis = image.load_img(number25bis, target_size=(32, 32))
number39bis = image.load_img(number39bis, target_size=(32, 32))


number3 = image.img_to_array(number3)
number9 = image.img_to_array(number9)
number39 = image.img_to_array(number39)
number17 = image.img_to_array(number17)
number31 = image.img_to_array(number31)

number13 = image.img_to_array(number13)
number19 = image.img_to_array(number19)
number25 = image.img_to_array(number25)
number25bis = image.img_to_array(number25bis)
number39bis = image.img_to_array(number39bis)


number3 /= 255.0
number9 /= 255.0
number39 /= 255.0
number17 /= 255.0
number31 /= 255.0

number13 /= 255.0
number19 /= 255.0
number25 /= 255.0
number25bis /= 255.0
number39bis /= 255.0

number3 = np.expand_dims(number3, axis=0)
number9 = np.expand_dims(number9, axis=0)
number39 = np.expand_dims(number39, axis=0)
number17 = np.expand_dims(number17, axis=0)
number31 = np.expand_dims(number31, axis=0)

number13 = np.expand_dims(number13, axis=0)
number19 = np.expand_dims(number19, axis=0)
number25 = np.expand_dims(number25, axis=0)
number25bis = np.expand_dims(number25bis, axis=0)
number39bis = np.expand_dims(number39bis, axis=0)

# Choose a model
model = load_model('./models/lenetpro.h5')
#model = load_model('./models/lenetplus.h5')
#model = load_model('./models/lenet.h5')

my3 = model.predict(number3)
my9 = model.predict(number9)
my39 = model.predict(number39)
my17 = model.predict(number17)
my31 = model.predict(number31)

other13 = model.predict(number13)
other19 = model.predict(number19)
other25 = model.predict(number25)
other251 = model.predict(number25bis)
other391 = model.predict(number39bis)

img3 = np.argmax(my3)
img9 = np.argmax(my9)
img39 = np.argmax(my39)
img17 = np.argmax(my17)
img31 = np.argmax(my31)

img13 = np.argmax(other13)
img19 = np.argmax(other19)
img25 = np.argmax(other25)
img251 = np.argmax(other251)
img392 = np.argmax(other391)

# My photos
print(f'Cartello 3: Classe predetta: {img3}')
print(f'Cartello 9: Classe predetta: {img9}')
print(f'Cartello 39: Classe predetta: {img39}')
print(f'Cartello 17: Classe predetta: {img17}')
print(f'Cartello 31: Classe predetta: {img31}')

# Other photos
print(f'Cartello 13: Classe predetta: {img13}')
print(f'Cartello 19: Classe predetta: {img19}')
print(f'Cartello 25 ruotato: Classe predetta: {img25}')
print(f'Cartello 25 orizzontale: Classe predetta: {img251}')
print(f'Cartello 39 new: Classe predetta: {img392}')

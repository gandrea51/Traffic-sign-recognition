from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np

def load_images(img_path, size=(32, 32)):
    '''Function per caricare e preprocessare una immagine'''

    img = image.load_img(img_path, target_size=size)
    img = image.img_to_array(img) / 255.0
    return np.expand_dims(img, axis=0)

def predictions(model, image_dictionary):
    '''Function per le previsioni su un dizionario'''

    results = {}
    for img_name, img_path in image_dictionary.items():
        img = load_images(img_path)
        pred = model.predict(img, verbose=0)
        results[img_name] = np.argmax(pred)
    return results

def shows(results):
    '''Function per i risultati'''

    for name, pred in results.items():
        print(f"{name}: Classe predetta {pred}")

def shows_predictions(image_dictionary, results):
    '''Function per mostrare le immagini con la previsione'''

    for name, path in image_dictionary.items():
        img = image.load_img(path)
        plt.imshow(img)
        plt.title(f"{name}: Classe predetta {results[name]}")
        plt.axis('off')
        plt.show()

# My photos
my_images = {
    "Cartello 3": "./images/Prova/3_personal.jpg",
    "Cartello 9": "./images/Prova/9_personal.jpg",
    "Cartello 39": "./images/Prova/39_personal.jpg",
    "Cartello 17": "./images/Prova/17_personal.jpg",
    "Cartello 31": "./images/Prova/31_personal.jpg"
}

# Other photos
other_images = {
    "Cartello 13": "./images/Prova/13_int.jpg",
    "Cartello 19": "./images/Prova/19_int.jpg",
    "Cartello 25 (doppio)": "./images/Prova/25_int.jpg",
    "Cartello 25": "./images/Prova/25_temp.jpeg",
    "Cartello 39": "./images/Prova/39_temp.jpg"
}

# Particular photos
particular_images = {
    "Cartello 1": "./other/sign_1.jpg",
    "Cartello 4": "./other/sign_4.jpg",
    "Cartello 9": "./other/sign_9.jpg",
    "Cartello 18": "./other/sign_18.jpg",
    "Cartello 25": "./other/sign_25.jpg",
    "Cartello 29": "./other/sign_29.jpg",
    "Cartello 30": "./other/sign_30.jpg",
    "Cartello 33": "./other/sign_33.jpg",
    "Cartello 40": "./other/sign_40.jpg"
}

model = load_model('./models/lenetpro.h5')
#model = load_model('./models/lenetplus.h5')
#model = load_model('./models/lenet.h5')
#model = load_model('./models/net.h5')

my_results = predictions(model, my_images)
other_results = predictions(model, other_images)
particular_results = predictions(model, particular_images)

#shows(my_results)
#shows(other_results)
#shows(particular_results)

#shows_predictions(my_images, my_results)
#shows_predictions(other_images, other_results)
shows_predictions(particular_images, particular_results)

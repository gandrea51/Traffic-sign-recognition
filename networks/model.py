from tensorflow.keras import models, layers

def TakeThat(input_shape, classes):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(6, kernel_size=(5, 5), activation='relu', padding='same'),
        layers.AveragePooling2D(pool_size=(2, 2), strides=2),
        layers.Conv2D(16, kernel_size=(5, 5), strides=1, activation='relu'),
        layers.AveragePooling2D(pool_size=(2, 2), strides=2),

        layers.Flatten(),
        layers.Dense(units=120, activation='relu'),
        layers.Dense(units=84, activation='relu'),
        layers.Dense(units=classes, activation='softmax')
    ])
    return model

def SpandauBallet(input_shape, classes):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        
        layers.Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2), strides=2),
        
        layers.Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2), strides=2),
        
        layers.Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2), strides=2),
        
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(classes, activation='softmax')
    ])
    return model

def Queen(input_shape, classes):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        
        layers.Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.AveragePooling2D(pool_size=(2, 2), strides=2),
        
        layers.Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.AveragePooling2D(pool_size=(2, 2), strides=2),
        
        layers.Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.AveragePooling2D(pool_size=(2, 2), strides=2),
        
        layers.Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.AveragePooling2D(pool_size=(2, 2), strides=2),

        layers.Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.AveragePooling2D(pool_size=(2, 2), strides=2),

        layers.GlobalAveragePooling2D(),        
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(classes, activation='softmax')
    ])
    return model
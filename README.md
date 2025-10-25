# 🛑 Traffic Sign Recognition - GTSRB Project

Project work by **Deep Learning** for the recognition of road signs using **GTSRB**.  
Three different **convolutional neural networks (CNNs)** based on architectures inspired by **LeNet5** were implemented and analyzed.

---

## 📌 Project objectives

The objective is to train and evaluate CNNs that manage to classify the **43 types of road signs** present in the **GTSRB** dataset, by means of

- Data preprocessing and **data augmentation**;
- **several CNN** development;
- Testing on **external images** to the dataset;
- Error and **generalization ability analysis**.

---

## 📂 How to consult the repository

The following structure shows the main contents of the repository:

```bash
GTSRB-Traffic-Sign-Recognition/
│
├── data/           # Original GTSRB dataset signals
├── dataset/        # Dataset preprocess: training, validation, testing
│   ├── X_train.npy / y_train.npy
│   ├── X_val.npy / y_val.npy
│   ├── X_test.npy / y_test.npy
│   ├── X_augmented.npy / y_augmented.npy
│
├── images/        # Images for external tests
│   ├── dataset/      # Signals before and after data augmentation
│   ├── summary/      # Detailed structure of networks (model.summary)
│   ├── test/
│   │   ├── internet/     # Tags cut from images found online
│   │   ├── original/     # Whole original images
│   │   ├── particular/   # Signs with particular fonts/shapes
│   │   ├── personal/     # Signs cut from personal photos
│   ├── training/     # Training progress and confusion matrices
│
├── list/           # List of signals and labels after preprocessing
├── models/         # CNN saved (.h5)
├── networks/       # Implementation of CNN architectures
├── scripts/        # Varietà di script per pre-elaborazione, test e aumento dei dati
│
├── GTSRB_Readme    # Complete project work documentation
├── LICENSE         # Do you want to use the project? Here are the rules
└── README.md       # This file
```
---

## 📦 Contents of the Repository

This repository **does not** include:
- The **GTSRB dataset** (you can download it from the [official site](https://benchmark.ini.rub.de/gtsrb_dataset.html)).
- The **trained CNN models (.h5)** due to size limitations.

It **does** include:
- 🧩 **CNN definitions** → `networks/` contains the code for all models (LeNet5, LeNet5Plus, LeNet5Pro, etc.).
- 🛠️ **Preprocessing & testing scripts** → `scripts/` contains the full pipeline for dataset preprocessing, augmentation, and testing.
- 🖼️ **External images for testing** → `images/` provides ready-to-use samples for evaluating the models.
- 📄 **Documentation** → `GTSRB_Readme.pdf` explains the entire workflow, models, and results.
- 📊 **PowerPoint presentation** → high-level summary of the project.

---

## 🚀 How to run the project

To run it, follow the steps below:

- Clone repository;
- Install all necessary dependencies;
- Download the dataset and Preprocess it;
- Start training by executing the command ```python networks/net_name.py``` and commenting on the lines of code as appropriate;
- (Optional) Perform external image tests with the command ```python scripts/test.py```.

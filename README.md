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

## 🚀 How to run the project

To run it, follow the steps below:

- Clone repository
- Install all necessary dependencies
- Start the training by executing the command ```python networks/lenet.py```
- Perform external image tests with the command ```python scripts/test.py```

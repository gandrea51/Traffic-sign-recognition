from trainer import loading, compiling, training, evaluating, plotting, confusioning
from models import TakeThat

def main():
    X_train, Y_train, X_val, Y_val, X_test, Y_test = loading("./dataset", 43)

    model = TakeThat((32, 32, 3), 43)
    model = compiling(model)
    model.summary()

    history = training(model, X_train, Y_train, X_val, Y_val)
    evaluating(model, X_test, Y_test)
    plotting(history)
    confusioning(model, X_test, Y_test)

    model.save("./model/TakeThat.h5")

if __name__ == "__main__":
    main()
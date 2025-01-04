from utils_and_constant import *
import json

from metrics_and_plots import plot_confusion_matrix, save_metrics
from model import evaluate_model, train_model




def main():
    X_train, y_train= load_data(PROCESSED_DATASET_TRAIN)
    X_test, y_test= load_data(PROCESSED_DATASET_TEST)

    # Train the model using the training set
    model = train_model(X_train, y_train)
    
    # Calculate test set metrics
    metrics = evaluate_model(model, X_test, y_test)

    print("====================Test Set Metrics==================")
    print(json.dumps(metrics, indent=2))
    print("======================================================")

    # Save metrics into json file
    save_metrics(metrics)
    plot_confusion_matrix(model, X_test, y_test)


if __name__ == "__main__":
    main()

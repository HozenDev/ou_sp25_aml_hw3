import matplotlib.pyplot as plt
import numpy as np
import os
import pickle

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from keras.saving import load_model
from hw3_base import load_precached_folds
from hw3_parser import create_parser
from core50 import Core50

#########################################
#             Load Results              #
#########################################

def load_trained_model(directory):
    """
    Load a trained Keras model from a given directory.
    :param directory: Path to the .keras model file
    :return: Loaded Keras model
    """
    return load_model(f"{directory}/model.keras")

def load_results(results_dir):
    results = []
    files = []
    for r_dir in results_dir:
        files.extend([os.path.join(r_dir, f) for f in os.listdir(r_dir) if f.endswith(".pkl")])

    for filename in files:
        with open(filename, "rb") as fp:
            data = pickle.load(fp)
            results.append(data)

    return results

#########################################
#             Plot Methods              #
#########################################

# Figure 3: Display sample test images with predicted probabilities
def plot_sample_predictions(dataset, model, class_names, num_samples=5):
    """
    Display sample images from the dataset along with model predictions.
    :param dataset: TensorFlow dataset object (test dataset)
    :param model: Trained model for prediction
    :param class_names: List of class names
    :param num_samples: Number of images to display
    """
    
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 5))

    images, labels = [], []
    for image_batch, label_batch in dataset.take(1):  # Take 1 batch
        images = image_batch.numpy()
        labels = label_batch.numpy()
        break  # Exit after first batch
    
    predictions = model.predict(images)
    predicted_labels = np.argmax(predictions, axis=1)
    
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 5))
    for i in range(num_samples):
        ax = axes[i]
        ax.imshow(images[i].astype("uint8"))
        title = f"True: {class_names[labels[i]]}\nPred: {class_names[predicted_labels[i]]}"
        ax.set_title(title)
        ax.axis("off")
    plt.show()

    
        
    """
    for i, ax in enumerate(axes):
        ax.imshow(images[i])
        predicted_label = np.argmax(predicted_probs[i])
        title = f"True: {class_names[true_labels[i]]}\nPred: {class_names[predicted_label]}"
        ax.set_title(title)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

    """

import numpy as np
import matplotlib.pyplot as plt

def compute_confusion_matrix(true_labels, predicted_labels, num_classes):
    """
    Compute the confusion matrix manually.
    
    :param true_labels: List or numpy array of true labels
    :param predicted_labels: List or numpy array of predicted labels
    :param num_classes: Number of unique classes
    :return: Confusion matrix as a numpy array
    """
    conf_matrix = np.zeros((num_classes, num_classes), dtype=int)

    for true, pred in zip(true_labels, predicted_labels):
        conf_matrix[true, pred] += 1  # Increase count in matrix

    return conf_matrix

    
# Figure 4a, 4b: Confusion matrix (aggregated across rotations)
def plot_confusion_matrix(true_labels_list, predicted_labels_list, class_names, title="Confusion Matrix"):
    cm = confusion_matrix(true_labels_list, predicted_labels_list)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap='Blues', values_format='d')
    plt.title(title)
    plt.show()

# Figure 5: Scatter plot of test accuracy for deep vs shallow models
def plot_accuracy_scatter(shallow_accuracies, deep_accuracies):
    plt.figure(figsize=(6, 6))
    plt.scatter(shallow_accuracies, deep_accuracies, label='Test Accuracy')
    plt.plot([0, 1], [0, 1], 'r--', label='y=x')
    plt.xlabel('Shallow Model Accuracy')
    plt.ylabel('Deep Model Accuracy')
    plt.title('Comparison of Shallow vs Deep Model Accuracies')
    plt.legend()
    plt.show()

#########################################
#            Main Function              #
#########################################
    
if __name__ == "__main__":
    # Use parser
    parser = create_parser()
    args = parser.parse_args()

    # if args.precache is None:
    #     # Load individual files (all objects); DON'T USE THIS CASE
    #     _, _, ds_testing, n_classes = load_data_set_by_folds(args, objects = list(range(10)))
    # else:
    #     # Load pre-cached data: this is what you want for HW 3
    #     _, _, ds_testing, n_classes = load_precached_folds(args)
    
    # Load data (modify paths accordingly)
    shallow_dir = ["./pkl/shallow_1/"]
    deep_dir = ["./pkl/default_deep/"]

    shallow_results = load_results(shallow_dir)
    # shallow_model = load_trained_model(shallow_dir)
    deep_results = load_results(deep_dir)
    # deep_model = load_trained_model(deep_dir)
    
    # Extract relevant data
    shallow_preds = [np.argmax(res['predict_testing'], axis=1) for res in shallow_results]
    deep_preds = [np.argmax(res['predict_testing'], axis=1) for res in deep_results]
    true_labels = range(0,5)  # Assuming same for all rotations

    print(deep_preds)
    print(true_labels)

    class_names = ['Plug Adapter', 'Scissors', 'Light Bulb', 'Cup']

    # Generate Figures
    # plot_sample_predictions(args, shallow_results[0]['predict_testing'], class_names)
    plot_confusion_matrix(true_labels, shallow_preds, class_names, "Shallow Model Confusion Matrix")
    plot_confusion_matrix(true_labels, deep_preds, class_names, "Deep Model Confusion Matrix")
    
    # Compute accuracy for each rotation
    # shallow_accuracies = [res['predict_testing_eval'][1] for res in shallow_results]
    # deep_accuracies = [res['predict_testing_eval'][1] for res in deep_results]
    # plot_accuracy_scatter(shallow_accuracies, deep_accuracies)
    

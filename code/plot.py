import matplotlib.pyplot as plt
import numpy as np
import os
import pickle

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from keras.saving import load_model
from hw3_base import load_precached_folds, check_args
from hw3_parser import create_parser

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

def plot_test_sample_with_predictions(test_ds, shallow_model_path, deep_model_path, num_samples=5):
    """
    Plots test images with probability distributions from the shallow and deep models.

    :param test_ds: TensorFlow dataset containing test images and labels
    :param shallow_model_path: Path to the trained shallow model
    :param deep_model_path: Path to the trained deep model
    :param num_samples: Number of test images to visualize
    """
    # Load trained models
    shallow_model = tf.keras.models.load_model(shallow_model_path)
    deep_model = tf.keras.models.load_model(deep_model_path)

    # Extract test samples
    images = []
    for image, _ in test_ds.take(1):
        images.append(image.numpy())
    
    num_samples = min(num_samples, len(images))
    shallow_predictions = shallow_model.predict(images[:num_samples])
    deep_predictions = deep_model.predict(images[:num_samples])

    class_names = ['Plug Adapter', 'Scissors', 'Light Bulb', 'Cup']  # Update if necessary

    fig, axes = plt.subplots(num_samples, 2, figsize=(12, 4 * num_samples))

    for i in range(num_samples):
        # Shallow model predictions
        axes[i, 0].imshow(images[i].astype("uint8"))
        axes[i, 0].axis('off')
        shallow_title = "\n".join([f"{class_names[j]}: {shallow_predictions[i][j]:.2f}" for j in range(len(class_names))])
        axes[i, 0].set_title(f"Shallow Model\n{shallow_title}", fontsize=10)

        # Deep model predictions
        axes[i, 1].imshow(images[i].astype("uint8"))
        axes[i, 1].axis('off')
        deep_title = "\n".join([f"{class_names[j]}: {deep_predictions[i][j]:.2f}" for j in range(len(class_names))])
        axes[i, 1].set_title(f"Deep Model\n{deep_title}", fontsize=10)

    plt.tight_layout()
    plt.show()

    
def plot_confusion_matrix(model_path, test_ds, title="Confusion Matrix"):
    """
    Computes and plots a confusion matrix for a given model and test dataset.

    :param model_path: Path to the trained model (.keras file)
    :param test_ds: TensorFlow dataset containing test images and labels
    :param title: Title of the confusion matrix plot
    """
    # Load the model
    model = tf.keras.models.load_model(model_path)

    # Extract labels and predictions
    y_true, y_pred = [], []
    for images, labels in test_ds:
        y_true.extend(labels.numpy())  # Convert tensors to numpy
        y_pred.extend(np.argmax(model.predict(images), axis=1))  # Predict classes

    # Compute and display confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    class_names = ['Plug Adapter', 'Scissors', 'Light Bulb', 'Cup']  # Adjust if needed

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap='Blues', values_format='d')
    plt.title(title)
    plt.show()

def plot_test_accuracy_scatter(shallow_results_files, deep_results_files):
    """
    Plots a scatter plot comparing test accuracies of shallow vs. deep models.

    :param shallow_results_files: List of paths to shallow model results .pkl files
    :param deep_results_files: List of paths to deep model results .pkl files
    """

    shallow_accuracies, deep_accuracies = [], []
    
    for result in shallow_results_files:
        shallow_accuracies.append(result["predict_testing_eval"][1])  # Extract accuracy

    for result in deep_results_files:
        deep_accuracies.append(result["predict_testing_eval"][1])  # Extract accuracy

    # Scatter plot
    plt.figure(figsize=(7,7))
    plt.scatter(shallow_accuracies, deep_accuracies, label="Runs")
    plt.plot([0, 1], [0, 1], 'k--', lw=2)  # y = x line

    plt.xlabel("Shallow Model Accuracy")
    plt.ylabel("Deep Model Accuracy")
    plt.title("Test Accuracy: Deep vs. Shallow")
    plt.legend()
    plt.grid(True)
    plt.show()


#########################################
#            Main Function              #
#########################################
    
if __name__ == "__main__":

    # Parse command-line arguments
    parser = create_parser()
    args = parser.parse_args()
    check_args(args)
    
    _, _, test_ds, _ = load_precached_folds(args)

    # Figure 3: Test Sample with Predictions
    # plot_test_sample_with_predictions(test_ds=test_ds, shallow_model_path="Net_Shallow_model.keras", deep_model_path="Net_Deep_model.keras", num_samples=5)

    # Figure 4a: Shallow Model Confusion Matrix
    # plot_confusion_matrix("Net_Shallow_model.keras", test_ds, title="Shallow Model Confusion Matrix")

    # Figure 4b: Deep Model Confusion Matrix
    # plot_confusion_matrix("Net_Deep_model.keras", test_ds, title="Deep Model Confusion Matrix")

    # Figure 5: Test Accuracy Scatter Plot
    shallow_results = load_results(["./pkl/shallow_1/"])
    deep_results = load_results(["./pkl/default_deep/"])

    plot_test_accuracy_scatter(shallow_results, deep_results)


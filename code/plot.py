import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import tensorflow as tf
import pandas as pd

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from hw3_base import load_precached_folds, check_args
from hw3_parser import create_parser

CORE50_METADATA_PATH = "/home/fagg/datasets/core50/core50_df.pkl"  # Adjust path if needed

#########################################
#             Class Mapping             #
#########################################

def get_class_mappings(core50_pkl_path, num_classes):
    """
    Loads core50_df.pkl and extracts object names for the trained classes.

    :param core50_pkl_path: Path to the core50 metadata pickle file.
    :param num_classes: Number of classes in the model's predictions
    :return: Dictionary mapping class indices (0 to num_classes-1) to object names
    """
    # Load the metadata file
    with open(core50_pkl_path, "rb") as f:
        df = pickle.load(f)

    # Extract unique class-object mappings
    class_mapping = df[['class', 'object']].drop_duplicates().sort_values(by='class')

    # Only keep the first `num_classes` entries (the ones used in training)
    trained_classes = class_mapping.iloc[:num_classes]

    # Create a dictionary mapping class index (0,1,2,3...) to human-readable object names
    class_dict = {i: f"Class {trained_classes.iloc[i]['class']} - Object {trained_classes.iloc[i]['object']}"
                  for i in range(num_classes)}

    return class_dict

#########################################
#             Load Results              #
#########################################

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

def plot_test_sample_with_predictions(test_ds, shallow_model_path, deep_model_path, num_samples=5, num_classes=4):
    """
    Plots test images with probability distributions from the shallow and deep models.

    :param test_ds: TensorFlow dataset containing test images and labels
    :param shallow_model_path: Path to the trained shallow model
    :param deep_model_path: Path to the trained deep model
    :param num_samples: Number of test images to visualize
    """
    # Load trained models
    shallow_model = tf.keras.models.load_model(shallow_model_path)
    # deep_model = tf.keras.models.load_model(deep_model_path)

    # Extract images and labels properly
    images = []
    for img_batch, _ in test_ds.take(num_samples):
        images.append(img_batch.numpy())  # Convert TensorFlow tensor to numpy

    # Convert lists to numpy arrays
    images = np.concatenate(images, axis=0)

    # Ensure we have the correct number of samples
    num_samples = min(num_samples, len(images))

    shallow_predictions = shallow_model.predict(images[:num_samples])
    # deep_predictions = deep_model.predict(images[:num_samples])

    # Convert images to uint8 for plotting
    images = (images * 255).astype(np.uint8)

    # Usage Example
    class_names = get_class_mappings(CORE50_METADATA_PATH, num_classes)

    # Fix issue when num_samples = 1
    fig, axes = plt.subplots(num_samples, 2, figsize=(12, 4 * num_samples))
    if num_samples == 1:
        axes = np.expand_dims(axes, axis=0)  # Ensure 2D shape

    for i in range(num_samples):
        # Ensure images are in correct format for `imshow`
        img = np.squeeze(images[i])  # Remove extra dimensions if needed

        # Shallow model predictions
        axes[i, 0].imshow(img.astype("uint8"))
        axes[i, 0].axis('off')
        shallow_title = "\n".join([f"{class_names[j]}: {shallow_predictions[i][j]:.2f}" for j in range(num_classes)])
        axes[i, 0].set_title(f"Shallow Model\n{shallow_title}", fontsize=10)

        # Deep model predictions
        axes[i, 1].imshow(img.astype("uint8"))
        axes[i, 1].axis('off')
        # deep_title = "\n".join([f"{class_names[j]}: {deep_predictions[i][j]:.2f}" for j in range(num_classes)])
        deep_title = "?"
        axes[i, 1].set_title(f"Deep Model\n{deep_title}", fontsize=10)

    plt.tight_layout()
    plt.savefig("figure_3.png")

    
def plot_confusion_matrix(model_path, test_ds, title="Confusion Matrix", filename="figure4.png", num_classes=4):
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
    class_names = get_class_mappings(CORE50_METADATA_PATH, num_classes)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(class_names.values()))
    disp.plot(cmap='Blues', values_format='d')
    plt.title(title)
    plt.savefig(filename)

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
    plt.savefig("figure_5.png")


#########################################
#            Main Function              #
#########################################
    
if __name__ == "__main__":

    # Parse command-line arguments
    parser = create_parser()
    args = parser.parse_args()
    check_args(args)
    
    _, _, test_ds, num_classes = load_precached_folds(args)

    shallow_model_dir ="./models/shallow_2/"
    shallow_model_name = f"{shallow_model_dir}image_Net_ShallowNet_Shallow_Csize_3_3_Cfilters_8_16_Pool_2_1_Pad_valid_hidden_32_drop_0.500_sdrop_0.200_L2_0.000100_LR_0.001000_ntrain_03_rot_00_model.keras"

    deep_model_dir = "./models/default_deep/"
    deep_model_name = f"{shallow_model_dir}image_Net_ShallowNet_Shallow_Csize_3_3_Cfilters_8_16_Pool_2_1_Pad_valid_hidden_32_drop_0.500_sdrop_0.200_L2_0.000100_LR_0.001000_ntrain_03_rot_00_model.keras"
    
    # Figure 3: Test Sample with Predictions
    plot_test_sample_with_predictions(test_ds=test_ds, shallow_model_path=shallow_model_name, deep_model_path=deep_model_name, num_samples=5, num_classes=num_classes)

    # Figure 4a: Shallow Model Confusion Matrix
    # plot_confusion_matrix(shallow_model_name, test_ds, title="Shallow Model Confusion Matrix", filename="figure_4a.png", num_classes=num_classes)

    # Figure 4b: Deep Model Confusion Matrix
    # plot_confusion_matrix(deep_model_name, test_ds, title="Deep Model Confusion Matrix", filename="figure_4b.png", num_classes=num_classes)

    # Figure 5: Test Accuracy Scatter Plot
    # shallow_results = load_results(["./pkl/shallow_1/"])
    # deep_results = load_results(["./pkl/default_deep/"])
    # plot_test_accuracy_scatter(shallow_results, deep_results)


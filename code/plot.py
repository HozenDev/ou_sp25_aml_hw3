import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pickle
import tensorflow as tf
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

#########################################
#             Load Results              #
#########################################

def load_results(filenames):
    results = []
    for filename in filenames:
        with open(filename, "rb") as f:
            results.append(pickle.load(f))
    return results

#########################################
#             Plot Methods              #
#########################################

# Figure 3: Display sample test images with predicted probabilities
def plot_sample_predictions(images, true_labels, predicted_probs, class_names, num_samples=5):
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 5))
    for i, ax in enumerate(axes):
        ax.imshow(images[i])
        predicted_label = np.argmax(predicted_probs[i])
        title = f"True: {class_names[true_labels[i]]}\nPred: {class_names[predicted_label]}"
        ax.set_title(title)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

# Figure 4a, 4b: Confusion matrix (aggregated across rotations)
def plot_confusion_matrix(true_labels_list, predicted_labels_list, class_names, title="Confusion Matrix"):
    all_true_labels = np.concatenate(true_labels_list)
    all_predicted_labels = np.concatenate(predicted_labels_list)
    cm = confusion_matrix(all_true_labels, all_predicted_labels)
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
    # Load data (modify paths accordingly)
    shallow_filenames = [f"shallow_model_rotation_{i}.pkl" for i in range(5)]
    deep_filenames = [f"deep_model_rotation_{i}.pkl" for i in range(5)]
    
    shallow_results = load_results(shallow_filenames)
    deep_results = load_results(deep_filenames)
    
    # Extract relevant data
    shallow_preds = [np.argmax(res['predict_testing'], axis=1) for res in shallow_results]
    deep_preds = [np.argmax(res['predict_testing'], axis=1) for res in deep_results]
    true_labels = [res['predict_testing_eval'][1] for res in shallow_results]  # Assuming same for all rotations
    
    class_names = ['Plug Adapter', 'Scissors', 'Light Bulb', 'Cup']

    # Generate Figures
    plot_sample_predictions(shallow_results[0]['predict_testing'], true_labels[0], shallow_results[0]['predict_testing'], class_names)
    plot_confusion_matrix(true_labels, shallow_preds, class_names, "Shallow Model Confusion Matrix")
    plot_confusion_matrix(true_labels, deep_preds, class_names, "Deep Model Confusion Matrix")
    
    # Compute accuracy for each rotation
    shallow_accuracies = [res['predict_testing_eval'][1] for res in shallow_results]
    deep_accuracies = [res['predict_testing_eval'][1] for res in deep_results]
    plot_accuracy_scatter(shallow_accuracies, deep_accuracies)
    

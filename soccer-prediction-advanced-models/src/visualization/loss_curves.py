import matplotlib.pyplot as plt

def plot_loss_curves(history):
    """
    Plots the training and validation loss curves.

    Parameters:
    history : dict
        A dictionary containing the training and validation loss values.
    """
    plt.figure(figsize=(12, 6))
    
    # Plot training & validation loss values
    plt.plot(history['loss'], label='Training Loss', color='blue')
    plt.plot(history['val_loss'], label='Validation Loss', color='orange')
    
    plt.title('Model Loss Curves')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.show()

def plot_multiple_loss_curves(histories, labels):
    """
    Plots loss curves for multiple models.

    Parameters:
    histories : list of dict
        A list of dictionaries containing the training and validation loss values for each model.
    labels : list of str
        A list of labels corresponding to each model.
    """
    plt.figure(figsize=(12, 6))
    
    for history, label in zip(histories, labels):
        plt.plot(history['loss'], label=f'{label} Training Loss')
        plt.plot(history['val_loss'], label=f'{label} Validation Loss', linestyle='--')
    
    plt.title('Comparison of Model Loss Curves')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.show()
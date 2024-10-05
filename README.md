# Brain Tumor MRI Classification Documentation

## Table of Contents
1. [Preprocessing Steps](#preprocessing-steps)
2. [Model Architecture](#model-architecture)
3. [Training Process](#training-process)
4. [Evaluation Metrics](#evaluation-metrics)
5. [Instructions](#instructions)

## Preprocessing Steps

### Data Loading
The dataset is loaded from multiple sources using the Hugging Face Datasets library. The datasets used include:
- [Simezu/brain-tumour-MRI-scan](https://huggingface.co/datasets/Simezu/brain-tumour-MRI-scan) - 5.71k training images, 1.31k test images.
- [PranomVignesh/MRI-Images-of-Brain-Tumor](https://huggingface.co/datasets/PranomVignesh/MRI-Images-of-Brain-Tumor) - 3.76k training images, 1.6k test images.
- [rhyssh/Brain-Tumor-MRI-Dataset-Training](https://huggingface.co/datasets/rhyssh/Brain-Tumor-MRI-Dataset-Training) - 800 training images.

### Label Harmonization
Labels are harmonized across datasets to create a unified labeling scheme. The labels `notumor` and `tumor` are mapped to 0 and 1, respectively.

### Removing Duplicates
The following steps are taken to remove duplicates:
1. Images are hashed using the average hash method from the `imagehash` library.
2. Duplicate images are identified and removed based on their hash values.
3. Any images present in both training and testing datasets are also removed from the training set.

### Data Conversion
The datasets are converted into Pandas DataFrames for easier manipulation and analysis.

### Data Augmentation and Normalization
The dataset is transformed for model input:
- **Resizing**: Images are resized to 224x224 pixels.
- **Normalization**: Images are normalized using the mean and standard deviation of the ImageNet dataset:
  - Mean: [0.485, 0.456, 0.406]
  - Std: [0.229, 0.224, 0.225]

## Model Architecture

### Pre-trained Model
The model used for classification is **ResNet50**, a widely adopted convolutional neural network architecture. 

### Modifications
The fully connected (fc) layer of the ResNet50 model is modified to match the number of classes in the dataset (i.e., `n_classes`), which is derived from the unique labels present in the training data.

## Training Process

### Hyperparameters
- **Batch Size**: 64
- **Learning Rate**: 0.0001
- **Number of Epochs**: 10

### Training Loop
1. The model is trained using the Adam optimizer.
2. Loss is calculated using Cross Entropy Loss.
3. After each epoch, the model is evaluated on the test set to monitor performance.
4. The best model (with the lowest test loss) is saved to disk for later use.

## Evaluation Metrics

### Initial Evaluation
Before training, the initial test loss and accuracy are calculated:
- **Initial Test Loss**: Calculated on the test dataset.
- **Initial Test Accuracy**: Percentage of correctly predicted labels.

### After Training
For each epoch, the following metrics are reported:
- **Train Loss**: Loss value on the training dataset.
- **Test Loss**: Loss value on the test dataset.
- **Test Accuracy**: Percentage of correctly predicted labels on the test dataset.

## Instructions

### Requirements
Ensure you have the following libraries installed:
```bash
pip install torch torchvision datasets imagehash matplotlib pandas
```

### Running the Code
1. Clone the repository:
```bash
git clone https://github.com/Espacio-root/Transfer-Learning-Tumor
```
2. Login to Hugging Face using the following command:
```bash
huggingface-cli login
```
3. Open `main.ipynb` in Jupyter Notebook or any other Python environment.
```bash
jupyter notebook main.ipynb
```
4. Run the cells in the notebook to execute the preprocessing, training, and evaluation steps.

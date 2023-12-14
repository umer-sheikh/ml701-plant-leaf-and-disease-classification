# Leaf Lens: Plant & Disease Classification

## Description
This project focuses on developing a machine learning model for plant and disease classification, utilizing a convolutional neural network (CNN). The model is trained to identify various plant species and detect common plant diseases through image analysis, applying advanced CNN architectures for accurate and efficient recognition.
## Installation

Before running the project, ensure that you have all the necessary libraries installed. You can install the dependencies using the following command:

```bash
pip install -r requirements.txt
```

## Dataset Structure

Download the dataset used in this project from the [New Plant Disease Dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset/data).

The dataset is organized within the `classification/dataset` directory, structured as follows:

```
classification/dataset
├── complete/
classification_cnn_dropout_hyperparamtuning.ipynb
classification_cnn_dropout.ipynb
classification_cnn.ipynb
requirements.txt
...
...
```

Make sure the data is arranged as shown above to properly run the model training and evaluation scripts.


## Checkpoints

Checkpoints are automatically saved in designated subdirectories within the `models` folder, corresponding to the configuration of your training session based on the dataset and feature tuning flags.
The pretrained model can be accessed here: https://drive.google.com/file/d/1HW5A2uyvv-KzxpE4kYTioqYxuWx_YEeP/view?usp=sharing

## Contributing

We welcome contributions to improve the model and its implementation. If you have suggestions or improvements, please open an issue to discuss your ideas before submitting a pull request.

## Contact

For any queries regarding this project, please open an issue in the GitHub repository, and we will get back to you.

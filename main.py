import os
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import tensorflow as tf
import numpy as np
from tensorflow import keras

png_paths = []
dataset = []
images = []
labels = []

# Download the dataset from the following link: https://www.kaggle.com/datasets/paultimothymooney/breast-histopathology-images
# Once the dataset is downloaded, move the folder called IDC_regular_ps50_idx5 from this dataset 
# to the current directory (move it to the same directory as this file)
# Loop through all the files in the folder IDC_regular_ps50_idx5 and add all the filepaths to the list png_paths
for root, _, filenames in os.walk("./IDC_regular_ps50_idx5"):
    for filename in filenames:
        if filename[-3:] == "png":
            filepath = root + "/" + filename
            png_paths.append(filepath)

# Create a labels list using the parent folders (0 or 1) as the labels for each image patch
# If the image patch label is 0, it is non-IDC and if the image patch label is 1, it is IDC
labels_list = [ (0.0, 1.0)[path.split('/')[-2] == '1'] for path in sorted(png_paths)]

# Create the training and validation dataset from the images in the IDC_regular_ps50_idx5 folder and the labels (stored in labels_list)
training_dataset, validation_dataset = tf.keras.utils.image_dataset_from_directory(
    directory = "./IDC_regular_ps50_idx5",
    labels = labels_list,
    label_mode = "binary",
    validation_split = 0.2,
    subset = "both",
    seed = 100,
    batch_size = 20000,
    image_size = (50, 50)
)

training_dataset = training_dataset.shuffle(1000).cache()
validation_dataset = validation_dataset.shuffle(1000).cache()

# Create convolutional neural network (CNN)
num_classes = 2
model = tf.keras.Sequential([
  tf.keras.layers.Rescaling(1./255, input_shape=(50, 50, 3)),
  tf.keras.layers.Conv2D(16, (3, 3), padding='same', activation='relu'),
  tf.keras.layers.MaxPooling2D(pool_size = (2, 2)),
  tf.keras.layers.Conv2D(16, (3, 3), padding='same', activation='relu'),
  tf.keras.layers.MaxPooling2D(pool_size = (2, 2)),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(32, activation='relu'),
  tf.keras.layers.Dense(num_classes)
])

# Create multi layer perceptron (MLP) which is called model_simple as it is a simpler model than the CNN
num_classes = 2
model_simple = tf.keras.Sequential([
  tf.keras.layers.Rescaling(1./255, input_shape=(50, 50, 3)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(num_classes)
])

# Compile the models
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model_simple.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Train the CNN model
model_checkpoint = tf.keras.callbacks.ModelCheckpoint('cnn_model/', save_best_only = True)
epochs = 40
history = model.fit(
  training_dataset,
  validation_data=validation_dataset,
  batch_size = 20000,
  epochs = epochs,
  callbacks = [model_checkpoint]
)

# Load trained CNN model
model = tf.keras.models.load_model('cnn_model/')

# Accuracy and loss plots for the CNN model - https://www.tensorflow.org/tutorials/images/classification
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()


# Train the MLP model
model_simple_checkpoint = tf.keras.callbacks.ModelCheckpoint('mlp_model/', save_best_only = True)
epochs = 40
history_simple = model_simple.fit(
  training_dataset,
  validation_data=validation_dataset,
  batch_size = 20000,
  epochs = epochs,
  callbacks = [model_simple_checkpoint]
)

# Load the MLP model
model_simple = tf.keras.models.load_model('mlp_model/')

# Accuracy and loss plots for the MLP model
acc = history_simple.history['accuracy']
val_acc = history_simple.history['val_accuracy']

loss = history_simple.history['loss']
val_loss = history_simple.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

# Confusion matrix for the CNN model
probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
test_images = [image for image, label in validation_dataset]
test_labels = [label for image, label in validation_dataset]
predictions_list = [np.argmax(prediction) for prediction in probability_model.predict(test_images[0])]
confusion_matrix = tf.math.confusion_matrix(labels = test_labels[0], predictions = predictions_list, num_classes = 2)
confusion_matrix = confusion_matrix.numpy()
fig_matrix, ax_matrix = plt.subplots()
image_matrix = ax_matrix.imshow(confusion_matrix)
ax_matrix.set_xticks(np.arange(2), labels = [0, 1])
ax_matrix.set_yticks(np.arange(2), labels = [0, 1])
ax_matrix.set_xlabel("Predicted label")
ax_matrix.set_ylabel("True label")
ax_matrix.set_title("Confusion matrix for CNN model")
for i in range(2):
    for j in range(2):
        if (confusion_matrix[i, j] >= 7000):
            text = ax_matrix.text(j, i, confusion_matrix[i, j], ha = "center", va = "center", color = "b")
        else:
            text = ax_matrix.text(j, i, confusion_matrix[i, j], ha = "center", va = "center", color = "w")
ax_matrix.figure.colorbar(image_matrix, ax = ax_matrix)

fig_matrix.tight_layout()
plt.show()

# Confusion matrix for the MLP model
probability_model_simple = tf.keras.Sequential([model_simple, tf.keras.layers.Softmax()])
test_images = [image for image, label in validation_dataset]
test_labels = [label for image, label in validation_dataset]
predictions_list = [np.argmax(prediction) for prediction in probability_model_simple.predict(test_images[0])]
confusion_matrix = tf.math.confusion_matrix(labels = test_labels[0], predictions = predictions_list, num_classes = 2)
confusion_matrix = confusion_matrix.numpy()
fig_matrix, ax_matrix = plt.subplots()
image_matrix = ax_matrix.imshow(confusion_matrix)
ax_matrix.set_xticks(np.arange(2), labels = [0, 1])
ax_matrix.set_yticks(np.arange(2), labels = [0, 1])
ax_matrix.set_xlabel("Predicted label")
ax_matrix.set_ylabel("True label")
ax_matrix.set_title("Confusion matrix for MLP model")
for i in range(2):
    for j in range(2):
        if (confusion_matrix[i, j] >= 7000):
            text = ax_matrix.text(j, i, confusion_matrix[i, j], ha = "center", va = "center", color = "b")
        else:
            text = ax_matrix.text(j, i, confusion_matrix[i, j], ha = "center", va = "center", color = "w")
ax_matrix.figure.colorbar(image_matrix, ax = ax_matrix)

fig_matrix.tight_layout()
plt.show()


# Test the models' predictions for one whole slide image by plotting the predictions 
# on a heatmap with different colors for true positive, true negative, false positive and false negative

# Loop through all the files in the folder IDC_regular_ps50_idx5/8863 (corresponding to all images for one whole slide image) 
# Add all the filepaths to the list png_paths_test
png_paths_test = []
for root, _, filenames in os.walk("./IDC_regular_ps50_idx5/8863"):
    for filename in filenames:
        if filename[-3:] == "png":
            filepath = root + "/" + filename
            png_paths_test.append(filepath)

# Create a labels list using the parent folders (0 or 1) as the labels for each image patch
# If the image patch label is 0, it is non-IDC and if the image patch label is 1, it is IDC
labels_list_test = [ (0.0, 1.0)[path.split('/')[-2] == '1'] for path in sorted(png_paths_test)]

# Create a dataset from the images in the IDC_regular_ps50_idx5/8863 folder 
# and the labels corresponding to the images in that folder (labels stored in labels_list_test)
testing_dataset = tf.keras.utils.image_dataset_from_directory(
    directory = "./IDC_regular_ps50_idx5/8863",
    labels = labels_list_test,
    label_mode = "binary",
    shuffle = False,
    batch_size = 1,
    image_size = (50, 50)
)


# Predictions on a whole slide image using the CNN model

# Add a softmax layer to the CNN model so it outputs the probability of the image patch 
# having a label of 0 and the probability of having a label of 1
probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])

test_images = [image for image, label in testing_dataset]
test_labels = [label for image, label in testing_dataset]

# Loop through all the images in the testing dataset (in the folder labelled 8863) 
# For each image inputted into the model, store the class that the model predicts (0 or 1 for non-IDC or IDC)
# by storing the class which has the maximum probability
predictions_list = []
for image in test_images:
    predictions_list.append(np.argmax(probability_model.predict(image)))

# Plot the predictions from the CNN model as a heatmap
fig, (ax, ax_simple) = plt.subplots(1, 2)
ax.set_ylim([0, 2000])
ax.set_xlim([0, 2000])
ax.set_title("Heatmap: CNN model")
for prediction, path in zip(predictions_list, sorted(png_paths_test)):
    y_value = float(path.split("_")[-2][1:])
    x_value = float(path.split("_")[-3][1:])
    label = float(path.split("/")[-2])
    image = plt.imread(path)
    if image.shape[0] != 50 or image.shape[1] != 50:
        continue
    ax.imshow(image, extent = [x_value, x_value + 50, y_value, y_value - 50])
    # True negative - blue
    if prediction == 0 and label == 0:
        ax.add_patch(Rectangle((x_value, y_value - 50), 50, 50, color = (0, 0, 1)))
    # False positive - yellow
    elif prediction == 1 and label == 0:
        ax.add_patch(Rectangle((x_value, y_value - 50), 50, 50, color = (1, 1, 0)))
    # False negative - red
    elif prediction == 0 and label == 1:
        ax.add_patch(Rectangle((x_value, y_value - 50), 50, 50, color = (1, 0, 0)))
    # True positive - green
    elif prediction == 1 and label == 1:
        ax.add_patch(Rectangle((x_value, y_value - 50), 50, 50, color = (0, 1, 0)))
plt.draw()


# Predictions on a whole slide image using the MLP model

# Add a softmax layer to the CNN model so it outputs the probability of the image patch 
# having a label of 0 and the probability of having a label of 1
probability_model_simple = tf.keras.Sequential([model_simple, tf.keras.layers.Softmax()])

# Loop through all the images in the testing dataset (in the folder labelled 8863) 
# For each image inputted into the model, store the class that the model predicts (0 or 1 for non-IDC or IDC)
# by storing the class which has the maximum probability
predictions_list_simple = []
for image in test_images:
    predictions_list_simple.append(np.argmax(probability_model_simple.predict(image)))

# Plot the predictions from the MLP model as a heatmap
ax_simple.set_ylim([0, 2000])
ax_simple.set_xlim([0, 2000])
ax_simple.set_title("Heatmap: MLP model")
for prediction, path in zip(predictions_list_simple, sorted(png_paths_test)):
    y_value = float(path.split("_")[-2][1:])
    x_value = float(path.split("_")[-3][1:])
    label = float(path.split("/")[-2])
    image = plt.imread(path)
    if image.shape[0] != 50 or image.shape[1] != 50:
        continue
    ax_simple.imshow(image, extent = [x_value, x_value + 50, y_value, y_value - 50])
    # True negative - blue
    if prediction == 0 and label == 0:
        ax_simple.add_patch(Rectangle((x_value, y_value - 50), 50, 50, color = (0, 0, 1)))
    # False positive - yellow
    elif prediction == 1 and label == 0:
        ax_simple.add_patch(Rectangle((x_value, y_value - 50), 50, 50, color = (1, 1, 0)))
    # False negative - red
    elif prediction == 0 and label == 1:
        ax_simple.add_patch(Rectangle((x_value, y_value - 50), 50, 50, color = (1, 0, 0)))
    # True positive - green
    elif prediction == 1 and label == 1:
        ax_simple.add_patch(Rectangle((x_value, y_value - 50), 50, 50, color = (0, 1, 0)))
plt.draw()
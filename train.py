import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
# DAY 2 -----
# --- 1. Define Constants & Directories ---
# Define image size. MobileNetV2 works well with 224x224
IMG_SIZE = (224, 224) 
BATCH_SIZE = 32 # Number of images to process in a batch

TRAIN_DIR = 'data/train'
VALID_DIR = 'data/validation'

# --- 2. Data Preprocessing and Augmentation ---
# Create an ImageDataGenerator for the training set.
# This will rescale pixels (normalization) and apply data augmentation.
# Data augmentation creates modified versions of your images (rotating, shifting, flipping)
# to help the model generalize better and prevent overfitting.
train_datagen = ImageDataGenerator(
    rescale=1./255.,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)

# For the validation set, we only need to rescale the pixels. No augmentation.
validation_datagen = ImageDataGenerator(rescale=1./255.)

# Create data generators that will read images from the directories
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary' # We have two classes: Good/Bad (or sharp/blurry)
)

validation_generator = validation_datagen.flow_from_directory(
    VALID_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary'
)
#DAY 3 -----
# --- 3. Model Building (Using Transfer Learning with MobileNetV2) ---
# Load the base MobileNetV2 model, pre-trained on the ImageNet dataset.
# We don't include the 'top' (the final classification layer) because we'll create our own.
base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')

# Freeze the layers of the base model. We don't want to retrain them.
base_model.trainable = False

# Add our custom classification layers on top of the base model
x = base_model.output
x = GlobalAveragePooling2D()(x) # Averages the spatial features
x = Dense(128, activation='relu')(x) # A fully-connected layer
# The final output layer has 1 neuron with a sigmoid activation function,
# which is perfect for binary (0 or 1) classification.
predictions = Dense(1, activation='sigmoid')(x) 

# Create the final model
model = Model(inputs=base_model.input, outputs=predictions)
#DAY 4 -----
# --- 4. Compile the Model ---
# We configure the model for training.
# 'adam' is an efficient optimizer.
# 'binary_crossentropy' is the standard loss function for two-class problems.
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print("Model Summary:")
model.summary()

# --- 5. Train the Model ---
# We'll train for 10 epochs. An epoch is one full pass through the entire training dataset.
EPOCHS = 10 

history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator
)

# --- 6. Save the Trained Model ---
# The model is saved in the HDF5 format.
model.save('image_quality_analyzer.h5')
print("\n--- Model training complete and saved as image_quality_analyzer.h5 ---")

# DAY 5 ----- (+ IT GIVE MATRIX AS WELL Figure_1.png)
# --- 7. Evaluate and Visualize Performance  ---
# Plotting the training and validation accuracy and loss helps us see how well the model learned.
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(EPOCHS)

plt.figure(figsize=(12, 5))
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
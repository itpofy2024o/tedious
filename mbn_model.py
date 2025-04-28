import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
import os
import sys
import shutil
import random
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import certifi
os.environ['SSL_CERT_FILE'] = certifi.where()

# Define image size and batch size
img_height, img_width = 224, 224
batch_size = 8  # Further reduced for 8GB RAM

def create_mobilenet_model(img_size, num_classes):
    """Creates a MobileNetV2 model with a custom classification head."""
    base_model = MobileNetV2(
        input_shape=(img_size, img_size, 3),
        include_top=False,
        weights='imagenet',
        pooling='avg'
    )
    base_model.trainable = False  # Freeze base model initially

    inputs = layers.Input(shape=(img_size, img_size, 3))
    x = base_model(inputs, training=False)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)  # High dropout to prevent overfitting
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs, outputs)
    return model, base_model

def prepare_validation_split(source_dir, val_dir, val_split=0.2):
    """Populate validation directory with class subfolders and images."""
    if not os.path.exists(val_dir):
        os.makedirs(val_dir)
    
    for folder in os.listdir(source_dir):
        folder_path = os.path.join(source_dir, folder)
        if os.path.isdir(folder_path):
            val_folder_path = os.path.join(val_dir, folder)
            if not os.path.exists(val_folder_path):
                os.makedirs(val_folder_path)
            
            images = [f for f in os.listdir(folder_path) if f.lower().endswith('.png')]
            random.shuffle(images)
            val_count = max(1, int(len(images) * val_split))  # At least 1 image
            for img in images[:val_count]:
                shutil.copy(os.path.join(folder_path, img), os.path.join(val_folder_path, img))
            print(f"Populated {val_folder_path} with {val_count} images")

if __name__ == "__main__":
    train_dir = sys.argv[1] if len(sys.argv) > 1 else './augmented'
    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"Directory {train_dir} does not exist.")
    val_dir = './val'
    if os.path.exists(val_dir):
        shutil.rmtree(val_dir)
    os.mkdir(val_dir)
    prepare_validation_split(train_dir, val_dir, val_split=0.2)
    
    # Validate directories
    for dir_path in [train_dir, val_dir]:
        class_folders = [f for f in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, f))]
        if not class_folders:
            raise ValueError(f"No class folders found in {dir_path}")
        for folder in class_folders:
            folder_path = os.path.join(dir_path, folder)
            images = [f for f in os.listdir(folder_path) if f.lower().endswith('.png')]
            if len(images) < batch_size:
                raise ValueError(f"{folder} in {dir_path} has {len(images)} images, need at least {batch_size}")

    # Dynamically determine number of classes
    num_classes = len([f for f in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, f))])
    print(f"Detected {num_classes} class folders in {train_dir}")

    # Verify dataset size
    expected_train_images_per_class = 1600  # 80% of 2000 (or 1200 for 1500/class)
    expected_val_images_per_class = 400    # 20% of 2000 (or 300 for 1500/class)
    for folder in os.listdir(train_dir):
        folder_path = os.path.join(train_dir, folder)
        if os.path.isdir(folder_path):
            num_images = len([f for f in os.listdir(folder_path) if f.lower().endswith('.png')])
            if num_images != expected_train_images_per_class:
                print(f"Warning: {folder} has {num_images} training images, expected {expected_train_images_per_class}")
    for folder in os.listdir(val_dir):
        folder_path = os.path.join(val_dir, folder)
        if os.path.isdir(folder_path):
            num_images = len([f for f in os.listdir(folder_path) if f.lower().endswith('.png')])
            if num_images != expected_val_images_per_class:
                print(f"Warning: {folder} has {num_images} validation images, expected {expected_val_images_per_class}")

    # Create data generators with minimal augmentation
    train_datagen = ImageDataGenerator(rescale=1./255)
    val_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True
    )

    validation_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )

    print("Class indices:", train_generator.class_indices)
    print(f"Training samples: {train_generator.samples}, Validation samples: {validation_generator.samples}")

    # Calculate steps explicitly
    steps_per_epoch = max(1, train_generator.samples // batch_size)
    validation_steps = max(1, validation_generator.samples // batch_size)
    print(f"Steps per epoch: {steps_per_epoch}, Validation steps: {validation_steps}")

    # Create and compile the model
    model, base_model = create_mobilenet_model(img_size=img_height, num_classes=num_classes)
    initial_learning_rate = 0.0005
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate, decay_steps=1000, decay_rate=0.9, staircase=True
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule, weight_decay=0.01),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary()

    # Define callbacks
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=10, restore_best_weights=True
    )
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        'best_mobilenet_model.keras', monitor='val_accuracy', save_best_only=True, mode='max'
    )

    # Initial training
    epochs = 50
    history = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        validation_data=validation_generator,
        validation_steps=validation_steps,
        epochs=epochs,
        callbacks=[early_stopping, checkpoint]
    )

    # Evaluate the model
    val_loss, val_accuracy = model.evaluate(validation_generator, steps=validation_steps)
    print(f"Initial Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

    # Fine-tuning if validation accuracy is below 60%
    if val_accuracy < 0.6:
        print("Fine-tuning MobileNetV2 layers...")
        base_model.trainable = True
        for layer in base_model.layers[:-20]:  # Unfreeze last 20 layers
            layer.trainable = False
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5, weight_decay=0.01),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        fine_tune_epochs = 10
        history_fine = model.fit(
            train_generator,
            steps_per_epoch=steps_per_epoch,
            validation_data=validation_generator,
            validation_steps=validation_steps,
            epochs=fine_tune_epochs,
            callbacks=[early_stopping, checkpoint]
        )
        for key in history.history:
            history.history[key].extend(history_fine.history[key])
        val_loss, val_accuracy = model.evaluate(validation_generator, steps=validation_steps)
        print(f"Fine-tuned Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

    # Plot training history (do not save to disk)
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    # Confusion matrix (do not save to disk)
    validation_generator.reset()
    y_pred = model.predict(validation_generator, steps=validation_steps)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = validation_generator.classes
    cm = confusion_matrix(y_true, y_pred_classes)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=validation_generator.class_indices.keys(), yticklabels=validation_generator.class_indices.keys())
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
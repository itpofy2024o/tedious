import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import certifi  # Added for SSL fix
os.environ['SSL_CERT_FILE'] = certifi.where()  # Set certificate path

# Define image size and batch size
img_height, img_width = 224, 224
batch_size = 32

def create_resnet_model(img_size, num_classes):
    """Creates a ResNet50 model with a custom classification head."""
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(img_size, img_size, 3))
    base_model.trainable = False

    inputs = layers.Input(shape=(img_size, img_size, 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs, outputs)
    return model, base_model

if __name__ == "__main__":
    train_dir = sys.argv[1] if len(sys.argv) > 1 else './augmented'
    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"Directory {train_dir} does not exist.")
    
    # Dynamically determine number of classes
    num_classes = len([f for f in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, f))])
    print(f"Detected {num_classes} class folders in {train_dir}")

    # Verify dataset size
    expected_images_per_class = 500
    for folder in os.listdir(train_dir):
        folder_path = os.path.join(train_dir, folder)
        if os.path.isdir(folder_path):
            num_images = len([f for f in os.listdir(folder_path) if f.lower().endswith('.png')])
            if num_images != expected_images_per_class:
                print(f"Warning: {folder} has {num_images} images, expected {expected_images_per_class}")

    # Create data generators with augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.1,
        zoom_range=0.15,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2,
        preprocessing_function=lambda x: tf.image.convert_image_dtype(x, tf.float32)
    )

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )

    validation_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )

    print("Class indices:", train_generator.class_indices)

    # Create and compile the model
    resnet_model, base_model = create_resnet_model(img_size=img_height, num_classes=num_classes)
    initial_learning_rate = 0.0001
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate, decay_steps=1000, decay_rate=0.9, staircase=True
    )
    resnet_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule, weight_decay=0.01),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    resnet_model.summary()

    # Define callbacks
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=10, restore_best_weights=True
    )
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        'best_resnet_model.keras', monitor='val_accuracy', save_best_only=True
    )

    # Initial training
    epochs = 50
    history = resnet_model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // train_generator.batch_size,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // train_generator.batch_size,
        epochs=epochs,
        callbacks=[early_stopping, checkpoint]
    )

    # Evaluate the model
    val_loss, val_accuracy = resnet_model.evaluate(validation_generator)
    print(f"Initial Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

    # Optional fine-tuning if validation accuracy is below 80%
    if val_accuracy < 0.8:
        print("Fine-tuning ResNet50 layers...")
        base_model.trainable = True
        for layer in base_model.layers[:-10]:
            layer.trainable = False
        resnet_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5, weight_decay=0.01),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        fine_tune_epochs = 10
        history_fine = resnet_model.fit(
            train_generator,
            steps_per_epoch=train_generator.samples // train_generator.batch_size,
            validation_data=validation_generator,
            validation_steps=validation_generator.samples // train_generator.batch_size,
            epochs=fine_tune_epochs,
            callbacks=[early_stopping, checkpoint]
        )
        for key in history.history:
            history.history[key].extend(history_fine.history[key])
        val_loss, val_accuracy = resnet_model.evaluate(validation_generator)
        print(f"Fine-tuned Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

    # Plot training history
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # Confusion matrix
    validation_generator.reset()
    y_pred = resnet_model.predict(validation_generator)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = validation_generator.classes
    cm = confusion_matrix(y_true, y_pred_classes)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=validation_generator.class_indices.keys(), yticklabels=validation_generator.class_indices.keys())
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

    # Save the final model
    resnet_model.save('resnet50_classifier.keras')
    print("Training finished. Final model saved as resnet50_classifier.keras")
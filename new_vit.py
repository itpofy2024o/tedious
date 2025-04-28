import shutil
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import certifi
os.environ['SSL_CERT_FILE'] = certifi.where()

# Define image size and batch size
img_height, img_width = 224, 224
batch_size = 32

def create_vit_model(img_size, num_classes):
    """Creates a lightweight Vision Transformer (ViT-Tiny) model."""
    def vit_tiny(input_shape, num_classes):
        inputs = layers.Input(shape=input_shape)
        # Patch embedding
        patch_size = 16
        num_patches = (img_size // patch_size) ** 2
        projection_dim = 192  # Reduced from 384
        x = layers.Conv2D(filters=projection_dim, kernel_size=patch_size, strides=patch_size, padding='valid')(inputs)
        x = layers.Reshape((num_patches, projection_dim))(x)
        # Positional embedding
        positions = tf.range(start=0, limit=num_patches, delta=1)
        pos_embedding = layers.Embedding(input_dim=num_patches, output_dim=projection_dim)(positions)
        x = x + pos_embedding
        # Transformer blocks
        num_heads = 4  # Reduced from 6
        for _ in range(8):  # Reduced from 12
            x = layers.LayerNormalization(epsilon=1e-6)(x)
            x = layers.MultiHeadAttention(num_heads=num_heads, key_dim=projection_dim // num_heads)(x, x)
            x = layers.Add()([x, x])
            x = layers.LayerNormalization(epsilon=1e-6)(x)
            x_ff = layers.Dense(projection_dim * 4, activation='gelu')(x)
            x_ff = layers.Dense(projection_dim)(x_ff)
            x = layers.Add()([x, x_ff])
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(num_classes, activation='softmax')(x)
        return models.Model(inputs, outputs)
    
    model = vit_tiny((img_size, img_size, 3), num_classes)
    return model, model

if __name__ == "__main__":
    # Use separate train and val directories
    train_dir = sys.argv[1] if len(sys.argv) > 1 else './augmented'
    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"Directory {train_dir} does not exist.")
    val_dir = './val'
    if os.path.exists(val_dir):
        shutil.rmtree(val_dir)
    os.mkdir(val_dir)
    
    # Dynamically determine number of classes
    num_classes = len([f for f in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, f))])
    print(f"Detected {num_classes} class folders in {train_dir}")

    # Verify dataset size (supports 500 or 1500 images per class)
    expected_train_images_per_class = 1200  # 80% of 1500 (or 400 for 500/class)
    expected_val_images_per_class = 300    # 20% of 1500 (or 100 for 500/class)
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
    vit_model, base_model = create_vit_model(img_size=img_height, num_classes=num_classes)
    initial_learning_rate = 0.0005
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate, decay_steps=1000, decay_rate=0.9, staircase=True
    )
    vit_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule, weight_decay=0.01),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    vit_model.summary()

    # Define callbacks
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=10, restore_best_weights=True
    )
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        'best_vit_model1.keras', monitor='val_accuracy', save_best_only=True, mode='max'
    )

    # Initial training
    epochs = 50
    history = vit_model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        validation_data=validation_generator,
        validation_steps=validation_steps,
        epochs=epochs,
        callbacks=[early_stopping, checkpoint]
    )

    # Evaluate the model
    val_loss, val_accuracy = vit_model.evaluate(validation_generator, steps=validation_steps)
    print(f"Initial Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

    # Fine-tuning if validation accuracy is below 50%
    if val_accuracy < 0.5:
        print("Fine-tuning ViT layers...")
        base_model.trainable = True
        for layer in base_model.layers[:50]:  # Freeze early layers (approx. first half)
            layer.trainable = False
        vit_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5, weight_decay=0.01),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        fine_tune_epochs = 10
        history_fine = vit_model.fit(
            train_generator,
            steps_per_epoch=steps_per_epoch,
            validation_data=validation_generator,
            validation_steps=validation_steps,
            epochs=fine_tune_epochs,
            callbacks=[early_stopping, checkpoint]
        )
        for key in history.history:
            history.history[key].extend(history_fine.history[key])
        val_loss, val_accuracy = vit_model.evaluate(validation_generator, steps=validation_steps)
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
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # Confusion matrix (do not save to disk)
    validation_generator.reset()
    y_pred = vit_model.predict(validation_generator, steps=validation_steps)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = validation_generator.classes
    cm = confusion_matrix(y_true, y_pred_classes)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=validation_generator.class_indices.keys(), yticklabels=validation_generator.class_indices.keys())
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
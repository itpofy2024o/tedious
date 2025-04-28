import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import sys
import matplotlib.pyplot as plt

# Define image size and number of classes
img_height, img_width = 224, 224

def create_vit_model(img_size, num_classes, patch_size=16, num_heads=4, embed_dim=128, num_transformer_blocks=4, mlp_dim=64, dropout_rate=0.3):
    """Creates a simplified Vision Transformer model."""
    inputs = layers.Input(shape=(img_size, img_size, 3))
    patches = layers.Conv2D(embed_dim, kernel_size=patch_size, strides=patch_size, padding='valid')(inputs)
    patches = layers.Reshape((-1, embed_dim))(patches)
    num_patches = (img_size // patch_size) ** 2
    positions = tf.range(start=0, limit=num_patches, delta=1)
    position_embeddings = layers.Embedding(input_dim=num_patches, output_dim=embed_dim)(positions)
    encoded_patches = patches + position_embeddings
    for _ in range(num_transformer_blocks):
        x = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        attention_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim // num_heads, dropout=dropout_rate)(x, x)
        x = layers.Add()([encoded_patches, attention_output])
        y = layers.LayerNormalization(epsilon=1e-6)(x)
        y = layers.Dense(mlp_dim, activation='gelu')(y)
        y = layers.Dropout(dropout_rate)(y)
        y = layers.Dense(embed_dim)(y)
        encoded_patches = layers.Add()([x, y])
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.GlobalAveragePooling1D()(representation)
    representation = layers.Dropout(dropout_rate)(representation)
    outputs = layers.Dense(num_classes, activation='softmax')(representation)
    model = models.Model(inputs=inputs, outputs=outputs)
    return model

if __name__ == "__main__":
    train_dir = sys.argv[1] if len(sys.argv) > 2 else 'augmented_portraits'
    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"Directory {train_dir} does not exist.")
    outname = sys.argv[2] if len(sys.argv) > 2 else 'new_model_trained'
    
    num_classes = len([f for f in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, f))])
    print(f"Detected {num_classes} class folders in {train_dir}")

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        horizontal_flip=True,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.2,
        shear_range=0.2,
        validation_split=0.2,
        preprocessing_function=lambda x: tf.image.convert_image_dtype(x, tf.float32)
    )

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(img_height, img_width),
        batch_size=16,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )

    validation_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(img_height, img_width),
        batch_size=16,
        class_mode='categorical',
        subset='validation',
        shuffle=False  # Disable shuffling for consistent evaluation
    )

    print("Class indices:", train_generator.class_indices)

    vit_model = create_vit_model(img_size=img_height, num_classes=num_classes, dropout_rate=0.3)
    initial_learning_rate = 0.0001
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate, decay_steps=1000, decay_rate=0.9, staircase=True
    )
    vit_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule, weight_decay=0.01),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    vit_model.summary()

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=10, restore_best_weights=True
    )
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        'best_vit_model{}.keras'.format(outname), monitor='val_accuracy', save_best_only=True
    )

    epochs = 50
    history = vit_model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // train_generator.batch_size,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // train_generator.batch_size,
        epochs=epochs,
        callbacks=[early_stopping, checkpoint]
    )

    val_loss, val_accuracy = vit_model.evaluate(validation_generator)
    print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

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

    vit_model.save('{}.keras'.format(outname))
    print("Training finished. Final model saved as human_portrait_classifier.keras")
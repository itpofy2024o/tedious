import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import sys

a= sys.argv[1]

def augment_images(input_dir, output_dir , images_per_class=500, target_size=(224, 224)):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    for person_folder in os.listdir(input_dir):
        person_path = os.path.join(input_dir, person_folder)
        if os.path.isdir(person_path):
            output_person_path = os.path.join(output_dir, person_folder)
            if not os.path.exists(output_person_path):
                os.makedirs(output_person_path)

            image_files = [f for f in os.listdir(person_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
            num_original_images = len(image_files)

            if num_original_images == 0:
                print(f"No images found in {person_folder}. Skipping...")
                continue

            augmentations_per_image = (images_per_class // num_original_images)
            extra_augmentations = images_per_class % num_original_images

            for idx, img_file in enumerate(image_files):
                img_path = os.path.join(person_path, img_file)
                img = tf.keras.preprocessing.image.load_img(img_path, target_size=target_size)
                x = tf.keras.preprocessing.image.img_to_array(img)
                x = x.reshape((1,) + x.shape)  # Add batch dimension

                # Determine how many augmentations for this image
                num_augs = augmentations_per_image + (1 if idx < extra_augmentations else 0)

                i = 0
                for batch in datagen.flow(
                    x,
                    batch_size=1,
                    save_to_dir=output_person_path,
                    save_prefix=f'{img_file.split(".")[0]}_augmented',
                    save_format='png'
                ):
                    i += 1
                    if i >= num_augs:
                        break

                # Copy the original image to ensure it's included
                # img.save(os.path.join(output_person_path, f'original_{img_file.split(".")[0]}.png'))

            print(f"Augmented {num_original_images} images for {person_folder} to reach {images_per_class} images.")

if __name__ == "__main__":
    input_image_directory = a  # Replace with the path to your main directory containing 11 folders
    augmented_image_directory = './augmented'  # Directory to save augmented images
    images_per_class = 500

    if os.path.exists(augmented_image_directory):
        os.system("rm -rf ${}".format(augmented_image_directory))

    # Create dummy folders and files for demonstration
    if not os.path.exists(input_image_directory):
        os.makedirs(input_image_directory)
        for i in range(1, os.listdir(input_image_directory).length):  # Create 11 dummy person folders
            person_folder = os.path.join(input_image_directory, f'person_{i}')
            os.makedirs(person_folder)
            num_images = 50 if i % 2 == 0 else 20  # Varying number of images
            for j in range(num_images):
                with open(os.path.join(person_folder, f'image_{j}.jpg'), 'w') as f:
                    f.write('')  # Create empty files

    augment_images(input_image_directory, augmented_image_directory, images_per_class=images_per_class)
    print(f"Augmented images saved to {augmented_image_directory}")

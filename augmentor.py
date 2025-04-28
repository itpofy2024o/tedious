import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import sys
import shutil
from PIL import Image
import random
import uuid

a = sys.argv[1]

def augment_images(input_dir, output_dir, images_per_class=500, target_size=(224, 224)):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.1,
        zoom_range=0.15,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    for person_folder in os.listdir(input_dir):
        person_path = os.path.join(input_dir, person_folder)
        if os.path.isdir(person_path):
            output_person_path = os.path.join(output_dir, person_folder)
            if not os.path.exists(output_person_path):
                os.makedirs(output_person_path)

            # Get list of PNG images
            image_files = [f for f in os.listdir(person_path) if f.lower().endswith('.png')]
            num_original_images = len(image_files)

            if num_original_images == 0:
                print(f"No images found in {person_folder}. Skipping...")
                continue

            print(f"{person_folder}: Found {num_original_images} original images")

            # Handle case where num_original_images >= images_per_class
            if num_original_images >= images_per_class:
                print(f"{person_folder} has {num_original_images} images, selecting {images_per_class}...")
                for idx, img_file in enumerate(image_files[:images_per_class]):
                    img_path = os.path.join(person_path, img_file)
                    try:
                        with Image.open(img_path) as img:
                            img.verify()
                        img = tf.keras.preprocessing.image.load_img(img_path, target_size=target_size)
                        img.save(os.path.join(output_person_path, f'original_{idx}.png'))
                    except Exception as e:
                        print(f"Error processing {img_file} in {person_folder}: {str(e)}")
                        continue
                num_generated = len([f for f in os.listdir(output_person_path) if f.lower().endswith('.png')])
                print(f"Augmented {num_original_images} images for {person_folder} to {num_generated}/{images_per_class} images.")
                continue

            # Copy all original images
            valid_image_files = []
            for idx, img_file in enumerate(image_files):
                img_path = os.path.join(person_path, img_file)
                try:
                    with Image.open(img_path) as img:
                        img.verify()
                    img = tf.keras.preprocessing.image.load_img(img_path, target_size=target_size)
                    img.save(os.path.join(output_person_path, f'original_{idx}.png'))
                    valid_image_files.append(img_file)
                except Exception as e:
                    print(f"Error processing {img_file} in {person_folder}: {str(e)}")
                    continue

            num_copied = len(valid_image_files)
            print(f"{person_folder}: Copied {num_copied} original images")

            # Calculate augmentations needed
            remaining_slots = images_per_class - num_copied
            total_augmentations = 0
            if remaining_slots > 0 and num_copied > 0:
                # Exact division to avoid overgeneration
                augmentations_per_image = remaining_slots // num_copied
                extra_augmentations = remaining_slots % num_copied  # Extra for first few images
                planned_augmentations = num_copied * augmentations_per_image + extra_augmentations

                print(f"{person_folder}: Generating {remaining_slots} augmentations ({augmentations_per_image} per image + {extra_augmentations} extra)")

                for idx, img_file in enumerate(valid_image_files):
                    img_path = os.path.join(person_path, img_file)
                    try:
                        img = tf.keras.preprocessing.image.load_img(img_path, target_size=target_size)
                        x = tf.keras.preprocessing.image.img_to_array(img)
                        x = x.reshape((1,) + x.shape)
                        num_augs = augmentations_per_image + (1 if idx < extra_augmentations else 0)
                        i = 0
                        while i < num_augs and total_augmentations < remaining_slots:
                            for batch in datagen.flow(
                                x,
                                batch_size=1,
                                save_to_dir=output_person_path,
                                save_prefix=f'augmented_{idx}_{uuid.uuid4().hex[:8]}',
                                save_format='png'
                            ):
                                i += 1
                                total_augmentations += 1
                                break  # Ensure one image per iteration
                        if total_augmentations >= remaining_slots:
                            break
                    except Exception as e:
                        print(f"Error augmenting {img_file} in {person_folder}: {str(e)}")
                        continue

                print(f"{person_folder}: Generated {total_augmentations} augmentations (planned: {planned_augmentations})")

            # Verify and adjust to exactly 500 images
            num_generated = len([f for f in os.listdir(output_person_path) if f.lower().endswith('.png')])
            if num_generated > images_per_class:
                print(f"{person_folder}: Generated {num_generated}, removing {num_generated - images_per_class} excess images...")
                # Sort files to keep originals and earliest augmentations
                all_files = [f for f in os.listdir(output_person_path) if f.lower().endswith('.png')]
                all_files.sort()  # Originals come first (original_0.png, etc.)
                excess_files = all_files[images_per_class:]  # Keep first 500
                for f in excess_files:
                    os.remove(os.path.join(output_person_path, f))
                num_generated = len([f for f in os.listdir(output_person_path) if f.lower().endswith('.png')])

            attempts = 0
            max_attempts = 10
            while num_generated < images_per_class and num_copied > 0 and attempts < max_attempts:
                print(f"{person_folder}: Generated {num_generated}, adding {images_per_class - num_generated} more...")
                for _ in range(images_per_class - num_generated):
                    img_file = random.choice(valid_image_files)
                    img_path = os.path.join(person_path, img_file)
                    try:
                        img = tf.keras.preprocessing.image.load_img(img_path, target_size=target_size)
                        x = tf.keras.preprocessing.image.img_to_array(img)
                        x = x.reshape((1,) + x.shape)
                        for batch in datagen.flow(
                            x,
                            batch_size=1,
                            save_to_dir=output_person_path,
                            save_prefix=f'augmented_extra_{uuid.uuid4().hex[:8]}',
                            save_format='png'
                        ):
                            break
                    except Exception as e:
                        print(f"Error in extra augmentation for {img_file} in {person_folder}: {str(e)}")
                        continue
                num_generated = len([f for f in os.listdir(output_person_path) if f.lower().endswith('.png')])
                attempts += 1

            if num_generated != images_per_class:
                print(f"Warning: {person_folder} has {num_generated}/{images_per_class} images after {attempts} attempts!")
            else:
                print(f"Augmented {num_copied} images for {person_folder} to {num_generated}/{images_per_class} images.")

if __name__ == "__main__":
    input_image_directory = a
    augmented_image_directory = './augmented'
    images_per_class = 500

    if os.path.exists(augmented_image_directory):
        shutil.rmtree(augmented_image_directory)

    if not os.path.exists(input_image_directory):
        raise FileNotFoundError(f"Input directory {input_image_directory} does not exist.")

    augment_images(input_image_directory, augmented_image_directory, images_per_class=images_per_class)
    print(f"Augmented images saved to {augmented_image_directory}")
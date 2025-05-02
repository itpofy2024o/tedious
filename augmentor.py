import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import sys
import shutil
from PIL import Image
import random
import uuid
import psutil
import random
import uuid
import psutil
import cv2
import mediapipe as mp
import numpy as np

mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

# Custom preprocessing function for contrast and saturation
def custom_preprocessing(image):
    # Random contrast adjustment
    contrast_factor = random.uniform(0.5, 1.5)  # Expanded range
    image = tf.image.adjust_contrast(image, contrast_factor)
    # Random saturation adjustment
    saturation_factor = random.uniform(0.5, 1.5)  # Expanded range
    image = tf.image.adjust_saturation(image, saturation_factor)
    return image

def check_disk_space(min_free_gb=5):
    """Check if enough disk space is available."""
    disk = psutil.disk_usage('/')
    free_gb = disk.free / (1024 ** 3)  # Convert bytes to GB
    if free_gb < min_free_gb:
        raise RuntimeError(f"Insufficient disk space: {free_gb:.2f}GB free, need at least {min_free_gb}GB.")
    print(f"Disk space check: {free_gb:.2f}GB free.")

def crop_face(image_path, target_size=(224, 224)):
    """Crop the face from an image using MediaPipe."""
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_detection.process(img_rgb)

    if results.detections:
        for detection in results.detections:
            bbox = detection.location_data.relative_bounding_box
            h, w, _ = img.shape
            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            width = int(bbox.width * w)
            height = int(bbox.height * h)

            # Ensure the bounding box is within image dimensions
            x = max(0, x)
            y = max(0, y)
            width = min(width, w - x)
            height = min(height, h - y)

            # Crop the face with some padding
            padding = 0.2  # 20% padding around the face
            x_padded = max(0, x - int(width * padding))
            y_padded = max(0, y - int(height * padding))
            width_padded = min(w - x_padded, int(width * (1 + 2 * padding)))
            height_padded = min(h - y_padded, int(height * (1 + 2 * padding)))

            face = img[y_padded:y_padded + height_padded, x_padded:x_padded + width_padded]
            if face.size == 0:
                return None

            # Resize to target size
            face = cv2.resize(face, target_size)

            # Convert back to PIL for saving
            face_pil = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
            return face_pil
    return None

def validate_image(image_path):
    """Validate if an image file is readable and non-corrupted."""
    try:
        with Image.open(image_path) as img:
            img.verify()
        return True
    except Exception:
        return False

def augment_images(input_dir, output_dir, images_per_class=1500, target_size=(224, 224)):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Check disk space before starting
    # check_disk_space(min_free_gb=2)

    datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=90,  # Increased for more posture variation
        width_shift_range=0.2,  # Increased for more shift
        height_shift_range=0.2,  # Increased for more shift
        shear_range=0.3,  # Increased for more distortion
        zoom_range=0.5,  # Increased for more zoom
        horizontal_flip=True,
        brightness_range=[0.5, 1.5],  # Added for lighting variation
        fill_mode='nearest',
        preprocessing_function=custom_preprocessing  # Contrast and saturation
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

            valid_image_files = []
            for idx, img_file in enumerate(image_files):
                img_path = os.path.join(person_path, img_file)
                try:
                    with Image.open(img_path) as img:
                        img.verify()
                    # Crop face using MediaPipe
                    cropped_img = crop_face(img_path, target_size)
                    if cropped_img is None:
                        print(f"No face detected in {img_file} in {person_folder}. Skipping...")
                        continue
                    cropped_img.save(os.path.join(output_person_path, f'cropped_{idx}.png'))
                    valid_image_files.append(img_file)
                except Exception as e:
                    print(f"Error processing {img_file} in {person_folder}: {str(e)}")
                    continue

            num_copied = len(valid_image_files)
            print(f"{person_folder}: Copied {num_copied} cropped original images")

            # Handle case where num_original_images >= images_per_class
            if num_copied >= images_per_class:
                print(f"{person_folder} has {num_copied} images, selecting {images_per_class}...")
                all_files = [f for f in os.listdir(output_person_path) if f.lower().endswith('.png')]
                all_files.sort()
                excess_files = all_files[images_per_class:]
                for f in excess_files:
                    os.remove(os.path.join(output_person_path, f))
                num_generated = len([f for f in os.listdir(output_person_path) if f.lower().endswith('.png')])
                print(f"Selected {num_copied} cropped images for {person_folder} to {num_generated}/{images_per_class} images.")
                continue

            # Calculate augmentations needed
            remaining_slots = images_per_class - num_copied
            total_augmentations = 0
            if remaining_slots > 0 and num_copied > 0:
                augmentations_per_image = remaining_slots // num_copied
                extra_augmentations = remaining_slots % num_copied
                planned_augmentations = num_copied * augmentations_per_image + extra_augmentations

                print(f"{person_folder}: Generating {remaining_slots} augmentations ({augmentations_per_image} per image + {extra_augmentations} extra)")

                # Process in smaller batches to reduce memory/disk usage
                batch_size = 100
                for idx, img_file in enumerate(valid_image_files):
                    img_path = os.path.join(person_path, img_file)
                    try:
                        # Load the cropped image instead of the original
                        cropped_img_path = os.path.join(output_person_path, f'cropped_{idx}.png')
                        img = tf.keras.preprocessing.image.load_img(cropped_img_path, target_size=target_size)
                        x = tf.keras.preprocessing.image.img_to_array(img)
                        x = x.reshape((1,) + x.shape)
                        num_augs = augmentations_per_image + (1 if idx < extra_augmentations else 0)
                        i = 0
                        temp_dir = os.path.join(output_person_path, 'temp')
                        if not os.path.exists(temp_dir):
                            os.makedirs(temp_dir)
                        while i < num_augs and total_augmentations < remaining_slots:
                            for batch in datagen.flow(
                                x,
                                batch_size=1,
                                save_to_dir=temp_dir,
                                save_prefix=f'augmented_{idx}_{uuid.uuid4().hex[:8]}',
                                save_format='png'
                            ):
                                aug_path = os.path.join(temp_dir, os.listdir(temp_dir)[0])
                                if not validate_image(aug_path):
                                    print(f"Generated augmented image {aug_path} is corrupted. Skipping...")
                                    os.remove(aug_path)
                                    continue
                                i += 1
                                total_augmentations += 1
                                break
                        # Move temp files to output directory and clean up
                        for temp_file in os.listdir(temp_dir):
                            shutil.move(os.path.join(temp_dir, temp_file), os.path.join(output_person_path, temp_file))
                        shutil.rmtree(temp_dir)
                        if total_augmentations >= remaining_slots:
                            break
                    except Exception as e:
                        print(f"Error augmenting {img_file} in {person_folder}: {str(e)}")
                        continue

                print(f"{person_folder}: Generated {total_augmentations} augmentations (planned: {planned_augmentations})")

            # Verify and adjust to exactly 1,500 images
            num_generated = len([f for f in os.listdir(output_person_path) if f.lower().endswith('.png')])
            if num_generated > images_per_class:
                print(f"{person_folder}: Generated {num_generated}, removing {num_generated - images_per_class} excess images...")
                all_files = [f for f in os.listdir(output_person_path) if f.lower().endswith('.png')]
                all_files.sort()
                excess_files = all_files[images_per_class:]
                for f in excess_files:
                    os.remove(os.path.join(output_person_path, f))
                num_generated = len([f for f in os.listdir(output_person_path) if f.lower().endswith('.png')])

            attempts = 0
            max_attempts = 10
            while num_generated < images_per_class and num_copied > 0 and attempts < max_attempts:
                print(f"{person_folder}: Generated {num_generated}, adding {images_per_class - num_generated} more...")
                temp_dir = os.path.join(output_person_path, 'temp')
                if not os.path.exists(temp_dir):
                    os.makedirs(temp_dir)
                for _ in range(images_per_class - num_generated):
                    img_file = random.choice(valid_image_files)
                    cropped_img_path = os.path.join(output_person_path, f'cropped_{valid_image_files.index(img_file)}.png')
                    try:
                        img = tf.keras.preprocessing.image.load_img(cropped_img_path, target_size=target_size)
                        x = tf.keras.preprocessing.image.img_to_array(img)
                        x = x.reshape((1,) + x.shape)
                        for batch in datagen.flow(
                            x,
                            batch_size=1,
                            save_to_dir=temp_dir,
                            save_prefix=f'augmented_extra_{uuid.uuid4().hex[:8]}',
                            save_format='png'
                        ):
                            aug_path = os.path.join(temp_dir, os.listdir(temp_dir)[0])
                            if not validate_image(aug_path):
                                print(f"Generated extra augmented image {aug_path} is corrupted. Skipping...")
                                os.remove(aug_path)
                                continue
                            break
                    except Exception as e:
                        print(f"Error in extra augmentation for {img_file} in {person_folder}: {str(e)}")
                        continue
                for temp_file in os.listdir(temp_dir):
                    shutil.move(os.path.join(temp_dir, temp_file), os.path.join(output_person_path, temp_file))
                shutil.rmtree(temp_dir)
                num_generated = len([f for f in os.listdir(output_person_path) if f.lower().endswith('.png')])
                attempts += 1

            if num_generated != images_per_class:
                print(f"Warning: {person_folder} has {num_generated}/{images_per_class} images after {attempts} attempts!")
            else:
                print(f"Augmented {num_copied} cropped images for {person_folder} to {num_generated}/{images_per_class} images.")

if __name__ == "__main__":
    input_image_directory = sys.argv[1]
    augmented_image_directory = './augmented'
    images_per_class = 2000

    if os.path.exists(augmented_image_directory):
        shutil.rmtree(augmented_image_directory)

    if not os.path.exists(input_image_directory):
        raise FileNotFoundError(f"Input directory {input_image_directory} does not exist.")

    augment_images(input_image_directory, augmented_image_directory, images_per_class=images_per_class)
    print(f"Augmented images saved to {augmented_image_directory}")
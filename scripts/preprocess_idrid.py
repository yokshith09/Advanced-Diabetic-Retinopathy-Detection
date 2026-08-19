import os
import glob
import cv2
import numpy as np
from tqdm import tqdm
import argparse

def crop_image_from_gray(img, tol=7):
    """
    Crop the black borders from the fundus image.
    """
    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1), mask.any(0))]
    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img > tol
        check_shape = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))].shape[0]
        if check_shape == 0: 
            return img 
        img1 = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
        img2 = img[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
        img3 = img[:, :, 2][np.ix_(mask.any(1), mask.any(0))]
        img = np.stack([img1, img2, img3], axis=-1)
        return img
    return img

def apply_gaussian_filter(image):
    """
    Apply Gaussian filter to enhance contrast (Ben Graham's method).
    """
    image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), 10), -4, 128)
    return image

def process_idrid(input_dir, output_dir, image_size=224):
    """
    Process IDRiD images (crop, resize, gaussian filter) to match APTOS small dataset.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all images in IDRiD B. Disease Grading (both train and test sets)
    # They are stored in A. Training Set and B. Testing Set
    image_paths = glob.glob(os.path.join(input_dir, "**", "*.jpg"), recursive=True)
    
    if not image_paths:
        print(f"No .jpg images found in {input_dir}.")
        return

    print(f"Found {len(image_paths)} images. Preprocessing to {image_size}x{image_size}...")
    
    for path in tqdm(image_paths):
        # Read image
        img = cv2.imread(path)
        if img is None:
            continue
            
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Crop borders
        img = crop_image_from_gray(img)
        
        # Resize
        img = cv2.resize(img, (image_size, image_size))
        
        # Apply Gaussian filter
        img = apply_gaussian_filter(img)
        
        # Save output
        filename = os.path.basename(path)
        out_path = os.path.join(output_dir, filename)
        
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(out_path, img)
        
    print(f"Done! Preprocessed images saved to {output_dir}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="data/raw/idrid/B. Disease Grading/1. Original Images")
    parser.add_argument("--output_dir", default="data/processed/idrid_224_gaussian")
    parser.add_argument("--image_size", type=int, default=224)
    args = parser.parse_args()
    
    process_idrid(args.input_dir, args.output_dir, args.image_size)

if __name__ == "__main__":
    main()

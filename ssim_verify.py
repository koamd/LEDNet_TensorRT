import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import argparse
import os

def list_image_files(folder_path, extensions=['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff']):
    """
    Lists all image files in the specified folder with given extensions.
    
    Parameters:
        folder_path (str): Path to the folder to search.
        extensions (list): List of valid image file extensions.
        
    Returns:
        List of image file paths.
    """
    image_files = [
        os.path.join(folder_path, f) 
        for f in os.listdir(folder_path) 
        if os.path.isfile(os.path.join(folder_path, f)) and f.lower().endswith(tuple(extensions))
    ]
    return image_files

def compute_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Computes the Structural Similarity Index (SSIM) between two images.
    """
    # Convert to grayscale if needed
    if len(img1.shape) == 3:
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    if len(img2.shape) == 3:
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    # Ensure same size
    if img1.shape != img2.shape:
        raise ValueError("Input images must have the same dimensions for SSIM.")

    score, _ = ssim(img1, img2, full=True)
    return score


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute SSIM between two image folder.")
    parser.add_argument("--img1_folder", type=str, help="Path to first image folder")
    parser.add_argument("--img2_folder", type=str, help="Path to second image folder")
    args = parser.parse_args()
    
    img1_folder = list_image_files(args.img1_folder)

    total_ssim = 0.0  

    # Load images
    for img1_file in img1_folder:

        filename = os.path.basename(img1_file)
        img1 = cv2.imread(img1_file)
        img2 = cv2.imread(os.path.join(args.img2_folder, filename))

        if img1 is None:
            raise FileNotFoundError(f"Could not load image: {img1_file}")
        if img2 is None:
            raise FileNotFoundError(f"Could not load image: {os.path.join(args.img2_folder, filename)}")

        score = compute_ssim(img1, img2)
        print(f"SSIM: {score:.4f}")

        total_ssim += score
    
    total_ssim /= len(img1_folder)
    print(f"Average SSIM: {total_ssim:.4f}")


#python ssim_verify.py --img1_folder './pytorch/uncompressed_ppm1/0052' --img2_folder './pytorch/uncompressed_ppm2/0052'
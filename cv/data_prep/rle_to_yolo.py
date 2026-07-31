import os
import shutil
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# NOTE ON CLASS NAMES:
# The Severstal Steel Defect Detection dataset has 4 anonymous defect classes (ClassId 1, 2, 3, 4).
# Per user constraints: We map these to neutral labels 'defect_class_1', 'defect_class_2', 
# 'defect_class_3', and 'defect_class_4' (which map to 0-indexed values 0, 1, 2, 3 in YOLO).
# Semantic names (corrosion, crack, pitting, weld-defect) will be attached in the 
# reasoning layer downstream rather than hardcoding assumptions at the CV level.

def rle_to_mask(rle_string, height=256, width=1600):
    """
    Decodes RLE string into a binary mask of shape (height, width).
    Severstal uses column-major order (top to bottom, then left to right).
    """
    if not rle_string or pd.isna(rle_string) or rle_string == '':
        return np.zeros((height, width), dtype=np.uint8)
    
    rle_numbers = [int(numstring) for numstring in rle_string.split(' ')]
    rle_pairs = np.array(rle_numbers).reshape(-1, 2)
    img = np.zeros(height * width, dtype=np.uint8)
    for start, length in rle_pairs:
        start -= 1
        img[start:start + length] = 1
    
    # Reshaping to (width, height) and transposing gives (height, width) with column-major indices
    return img.reshape(width, height).T

def mask_to_yolo_polygons(mask):
    """
    Finds contours in a binary mask and returns normalized polygon coordinates.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    h, w = mask.shape
    for contour in contours:
        # Simplify contour to reduce output label size
        epsilon = 0.001 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        pts = approx.reshape(-1, 2)
        if len(pts) < 3:
            continue
            
        polygon = []
        for x, y in pts:
            # Normalize to 0-1 range
            norm_x = x / float(w)
            norm_y = y / float(h)
            polygon.append((norm_x, norm_y))
        polygons.append(polygon)
    return polygons

def prepare_severstal_yolo(data_dir, output_dir, split_ratio=0.1, seed=42):
    """
    Reads train.csv, decodes masks, performs split, and sets up YOLO folders.
    """
    print(f"Loading CSV from {os.path.join(data_dir, 'train.csv')}...")
    csv_path = os.path.join(data_dir, 'train.csv')
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"train.csv not found at {csv_path}. Please place the dataset in {data_dir}")
        
    df = pd.read_csv(csv_path)
    
    # Handle CSV format differences:
    # Older Kaggle format: ImageId_ClassId column
    # Newer Kaggle format: ImageId, ClassId, EncodedPixels columns
    if 'ImageId_ClassId' in df.columns:
        df['ImageId'] = df['ImageId_ClassId'].apply(lambda x: x.split('_')[0])
        df['ClassId'] = df['ImageId_ClassId'].apply(lambda x: int(x.split('_')[1]))
    
    # Remove rows with empty RLEs to identify images with actual defects
    df_defects = df[df['EncodedPixels'].notna() & (df['EncodedPixels'] != '')].copy()
    
    # Map 1-4 to 0-3 (0-indexed for YOLO)
    df_defects['ClassId'] = df_defects['ClassId'].astype(int) - 1
    
    # Get list of unique images containing defects
    defect_images = df_defects['ImageId'].unique()
    
    # Train/Val split on unique images containing defects to prevent data leakage
    train_imgs, val_imgs = train_test_split(defect_images, test_size=split_ratio, random_state=seed)
    train_set = set(train_imgs)
    val_set = set(val_imgs)
    
    # Create output directories
    for split in ['train', 'val']:
        os.makedirs(os.path.join(output_dir, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'labels', split), exist_ok=True)
        
    source_img_dir = os.path.join(data_dir, 'train_images')
    if not os.path.exists(source_img_dir):
        raise FileNotFoundError(f"train_images directory not found at {source_img_dir}")

    # Process images and write labels
    print("Converting RLE masks to YOLO polygons and copying images...")
    grouped = df_defects.groupby('ImageId')
    
    for img_id, group in tqdm(grouped):
        if img_id in train_set:
            split = 'train'
        elif img_id in val_set:
            split = 'val'
        else:
            continue
            
        # Copy image file
        src_path = os.path.join(source_img_dir, img_id)
        dst_path = os.path.join(output_dir, 'images', split, img_id)
        if os.path.exists(src_path):
            shutil.copy(src_path, dst_path)
        else:
            print(f"Warning: Image file {src_path} not found.")
            continue
            
        # Write labels to txt file
        label_file = os.path.splitext(img_id)[0] + '.txt'
        label_path = os.path.join(output_dir, 'labels', split, label_file)
        
        with open(label_path, 'w') as f:
            for idx, row in group.iterrows():
                class_idx = row['ClassId']
                rle = row['EncodedPixels']
                
                mask = rle_to_mask(rle)
                polygons = mask_to_yolo_polygons(mask)
                
                for poly in polygons:
                    poly_str = " ".join([f"{x:.6f} {y:.6f}" for x, y in poly])
                    f.write(f"{class_idx} {poly_str}\n")
                    
    print(f"YOLO Dataset preparation completed at {output_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Convert Severstal RLE dataset to YOLO format")
    parser.add_argument("--data-dir", type=str, default="data/severstal", help="Raw Kaggle dataset path")
    parser.add_argument("--output-dir", type=str, default="data/severstal_yolo", help="YOLO output path")
    parser.add_argument("--split", type=float, default=0.1, help="Validation split size")
    args = parser.parse_args()
    
    prepare_severstal_yolo(args.data_dir, args.output_dir, args.split)

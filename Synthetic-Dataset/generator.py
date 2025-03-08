import os
import random
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import argparse

def load_images_from_folder(folder_path, limit=None):
    """Load all images from a folder."""
    images = []
    filenames = []
    i = 0
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(folder_path, filename)
            try:
                img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
                if img is not None:
                    images.append(img)
                    filenames.append(filename)
                    i+=1
                    if limit is not None and i >= limit:
                        break
            except Exception as e:
                print(f"Error loading {filename}: {e}")
    return images, filenames

def generate_random_homography(img_shape, max_translation=0.2, max_rotation=270, max_scale=1.1, max_shear=0.5, max_perspective=0.001):
    """
    Generate a realistic random homography matrix by perturbing translation, rotation,
    scale, shear, and perspective distortions within reasonable limits.
    """
    h, w = img_shape[:2]
    
    # Define small perturbations for each component
    max_translation = max_translation * min(h, w)  # Pixels
    max_rotation = np.radians(max_rotation)  # Degrees to radians
    max_shear = np.radians(max_shear)
    max_perspective = max_perspective 
    
    # Translation (tx, ty)
    tx = np.random.uniform(-max_translation, max_translation) + w / 2
    ty = np.random.uniform(-max_translation, max_translation) + h / 2
    T = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]], dtype=np.float32)
    
    # Rotation
    theta = np.random.uniform(-max_rotation, max_rotation)
    R = np.array([[np.cos(theta), -np.sin(theta), 0],
                  [np.sin(theta), np.cos(theta), 0],
                  [0, 0, 1]], dtype=np.float32)
    
    # Scaling
    sx = np.random.uniform(1 / max_scale, max_scale)
    sy = np.random.uniform(1 / max_scale, max_scale)
    S = np.array([[sx, 0, 0], [0, sy, 0], [0, 0, 1]], dtype=np.float32)
    
    # Shear
    shear_x = np.random.uniform(-max_shear, max_shear)
    shear_y = np.random.uniform(-max_shear, max_shear)
    Sh = np.array([[1, np.tan(shear_x), 0], [np.tan(shear_y), 1, 0], [0, 0, 1]], dtype=np.float32)
    
    # Perspective Distortion
    p1, p2 = np.random.uniform(-max_perspective, max_perspective, 2)
    P = np.array([[1, 0, 0], [0, 1, 0], [p1, p2, 1]], dtype=np.float32)
    
    # Final Homography: Apply transformations in a reasonable order
    H = P @ T @ Sh @ S @ R
    
    return H

def apply_homography(template, mask, H, output_shape):
    """Apply homography transformation to template and its mask."""
    h, w = template.shape[:2]
    
    # Apply homography to the template
    warped_template = cv2.warpPerspective(template, H, (output_shape[1], output_shape[0]))
    
    # Apply same homography to the mask
    if mask is None:
        # If no alpha channel, create a white mask of the same shape
        mask = np.ones((h, w), dtype=np.uint8) * 255
    warped_mask = cv2.warpPerspective(mask, H, (output_shape[1], output_shape[0]))
    
    return warped_template, warped_mask

def random_position(bg_shape, template_shape):
    """Generate a random position for the template within the background."""
    max_x = max(1, bg_shape[1] - template_shape[1])
    max_y = max(1, bg_shape[0] - template_shape[0])
    x = random.randint(0, max_x - 1)
    y = random.randint(0, max_y - 1)
    return x, y

def place_template(background, template, mask, position):
    """Place the template on the background at the specified position."""
    result = background.copy()
    x, y = position
    h, w = template.shape[:2]
    
    # Ensure we're within bounds
    if y + h > result.shape[0] or x + w > result.shape[1]:
        h = min(h, result.shape[0] - y)
        w = min(w, result.shape[1] - x)
    
    # If template has alpha channel, use it for blending
    if template.shape[2] == 4:
        alpha = template[:h, :w, 3] / 255.0
        alpha = np.expand_dims(alpha, axis=2)
        
        # Apply blending
        for c in range(3):
            result[y:y+h, x:x+w, c] = (1 - alpha[:, :, 0]) * result[y:y+h, x:x+w, c] + alpha[:, :, 0] * template[:h, :w, c]
    else:
        # Use mask for blending
        if mask is not None:
            alpha = mask[:h, :w] / 255.0
            for c in range(3):
                result[y:y+h, x:x+w, c] = (1 - alpha) * result[y:y+h, x:x+w, c] + alpha * template[:h, :w, c]
        else:
            # Simple overlay without alpha blending
            result[y:y+h, x:x+w] = template[:h, :w, :3]
    
    return result

def generate_dataset(minibatch, templates_folder, backgrounds_folder, output_folder, num_images):
    """Generate synthetic dataset with templates on backgrounds."""
    # Create output folders
    os.makedirs(output_folder, exist_ok=False)
    os.makedirs(os.path.join(output_folder, "images"), exist_ok=False)
    os.makedirs(os.path.join(output_folder, "masks"), exist_ok=False)
    
    # Load templates and backgrounds
    templates, template_filenames = load_images_from_folder(templates_folder, limit=2 if minibatch else None)
    backgrounds, bg_filenames = load_images_from_folder(backgrounds_folder, limit=2 if minibatch else None)
    
    print(f"Loaded {len(templates)} templates and {len(backgrounds)} backgrounds.")
    
    # Dataset info
    dataset_info = {
        "images": [],
        "templates_used": []
    }

    if minibatch:
        num_images = 4
    
    for i in tqdm(range(num_images), desc="Generating images"):
        # Select a random background
        bg_idx = random.randint(0, len(backgrounds) - 1)
        background = backgrounds[bg_idx].copy()
        
        # Convert to RGB if grayscale
        if len(background.shape) == 2:
            background = cv2.cvtColor(background, cv2.COLOR_GRAY2BGR)
        elif background.shape[2] == 4:
            background = background[:, :, :3]  # Remove alpha channel
        
        # Determine number of templates for this image (k ~ N(1, 1.5))
        # k = max(1, int(abs(np.random.normal(1, 1.5))))
        k = 1 # For now, just use 1 template per image
        
        image_info = {
            "image_id": i,
            "background": bg_filenames[bg_idx],
            "templates": []
        }
        
        # For each template
        for j in range(k):
            # Select a random template
            template_idx = random.randint(0, len(templates) - 1)
            template = templates[template_idx].copy()
            template_filename = template_filenames[template_idx]
            
            # Extract alpha channel as mask if it exists
            if template.shape[2] == 4:
                mask = template[:, :, 3]
                template_rgb = template[:, :, :3]
            else:
                mask = None
                template_rgb = template
            
            # Generate random homography
            H = generate_random_homography(background.shape)
            
            # Apply homography to template and mask
            warped_template, warped_mask = apply_homography(template_rgb, mask, H, background.shape)
            
            # Create full mask image (black with white template)
            full_mask = np.zeros((background.shape[0], background.shape[1]), dtype=np.uint8)
            
            # Find non-zero pixels in warped mask
            if warped_mask is not None:
                # Set those pixels to white in the full mask
                full_mask[warped_mask > 0] = 255
            
            # Apply the warped template to the background
            combined_img = background.copy()
            for h in range(background.shape[0]):
                for w in range(background.shape[1]):
                    if warped_mask is not None and warped_mask[h, w] > 0:
                        alpha = warped_mask[h, w] / 255.0
                        combined_img[h, w] = (1 - alpha) * background[h, w] + alpha * warped_template[h, w]
            
            # Save the individual mask
            mask_path = os.path.join(output_folder, "masks", f"img_{i:06d}_template_{j:02d}.png")
            cv2.imwrite(mask_path, full_mask)
            
            # # Save a separate image with just this template
            # separate_img_path = os.path.join(output_folder, "images", f"img_{i:06d}_template_{j:02d}.png")
            # cv2.imwrite(separate_img_path, combined_img)
            
            # Update background for the next template
            background = combined_img
            
            # Record information
            template_info = {
                "template_id": j,
                "template_file": template_filename,
                "homography_matrix": H.tolist(),
                "mask_file": f"masks/img_{i:06d}_template_{j:02d}.png",
                "image_file": f"images/img_{i:06d}_template_{j:02d}.png"
            }
            image_info["templates"].append(template_info)
        
        # Save the final combined image with all templates
        final_img_path = os.path.join(output_folder, "images", f"img_{i:06d}_combined.png")
        cv2.imwrite(final_img_path, background)
        
        image_info["combined_image_file"] = f"images/img_{i:06d}_combined.png"
        dataset_info["images"].append(image_info)
    
    # Save dataset info
    with open(os.path.join(output_folder, "dataset_info.json"), "w") as f:
        json.dump(dataset_info, f, indent=2)
    
    print(f"Generated {num_images} images with a total of {sum(len(img['templates']) for img in dataset_info['images'])} templates.")
    print(f"Output saved to {output_folder}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic dataset for template matching")
    parser.add_argument("--minibatch", action=argparse.BooleanOptionalAction, default=False, help="If true only loads 2 images and templates and generates 4 samples")
    parser.add_argument("--templates", type=str, required=True, help="Path to folder containing template images")
    parser.add_argument("--backgrounds", type=str, required=True, help="Path to folder containing background images")
    parser.add_argument("--output", type=str, required=True, help="Path to output folder")
    parser.add_argument("--num-images", type=int, default=1000, help="Number of images to generate")
    
    args = parser.parse_args()
    
    generate_dataset(args.minibatch, args.templates, args.backgrounds, args.output, args.num_images)
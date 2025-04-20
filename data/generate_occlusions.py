import cv2
import numpy as np
import os
from glob import glob
from PIL import Image

def overlay_image(bg, accessory, position):
    bg = bg.convert('RGBA')
    accessory = accessory.resize(position[2], Image.ANTIALIAS).convert('RGBA')
    x, y = position[0], position[1]
    
    new_img = Image.new("RGBA", bg.size)
    new_img.paste(accessory, (x, y), accessory)
    result = Image.alpha_composite(bg, new_img)
    return result.convert('RGB')

def generate_occluded_dataset(face_dir, accessory_dir, save_dir, occlusion_types):
    os.makedirs(save_dir, exist_ok=True)
    face_paths = glob(f"{face_dir}/*.jpg")

    for face_path in face_paths:
        face = Image.open(face_path).resize((256, 256))
        base_name = os.path.basename(face_path)
        for occ_type in occlusion_types:
            acc_path = os.path.join(accessory_dir, occ_type)
            acc_img = Image.open(np.random.choice(glob(acc_path + "/*.png")))
            
            # Custom positioning per accessory type
            if occ_type == 'mask':
                position = (70, 130, (120, 70))
            elif occ_type == 'sunglasses':
                position = (70, 85, (120, 40))
            elif occ_type == 'hat':
                position = (45, 15, (160, 90))
            else:
                continue

            occluded = overlay_image(face.copy(), acc_img, position)
            save_path = os.path.join(save_dir, f"{occ_type}_{base_name}")
            occluded.save(save_path)

# Example usage:
# generate_occluded_dataset('images', 'Accessories', 'images_occluded', ['mask', 'sunglasses', 'hat'])

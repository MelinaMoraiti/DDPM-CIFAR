import json
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import matplotlib.pyplot as plt

# Define augmentation pipeline
transform = A.Compose([
    # Resizing and cropping
    A.RandomResizedCrop(width=512, height=512, scale=(0.8, 1.0)),
    
    # Flipping
    A.HorizontalFlip(p=0.5),
    
    # Color jittering
    A.RandomBrightnessContrast(p=0.2),
    A.HueSaturationValue(p=0.2),
    A.RGBShift(p=0.2),
    
    # Blurring
    A.GaussianBlur(p=0.1),
    
    # Random erasing
    A.CoarseDropout(max_holes=2, max_height=16, max_width=16, fill_value=0, p=0.2),
    
    # Convert to tensor
    ToTensorV2()
])

# Load JSON mapping of class labels to flower names
with open('C:/Users/user/Downloads/102 flower/cat_to_name.json', 'r') as f:
    class_labels = json.load(f)

image_dir = '56'
image_file = 'image_02753.jpg'

# Load the example image
image_path = f'C:/Users/user/Downloads/102 flower/flowers/train/{image_dir}/{image_file}'
image = cv2.imread(image_path)

# Get the class label for the image
class_label = image_dir
class_name = class_labels.get(class_label)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

# Apply transformations
augmented = transform(image=image)
augmented_image = augmented["image"]

# Convert tensor to numpy array
augmented_image_np = augmented_image.permute(1, 2, 0).cpu().numpy()

# Display the original and augmented images
fig, ax = plt.subplots(1, 2, figsize=(15, 7))
ax[0].imshow(image)
ax[0].set_title(f'Original Image ({class_name})')
ax[0].axis('off')

ax[1].imshow(augmented_image_np)
ax[1].set_title(f'Augmented Image ({class_name})')
ax[1].axis('off')

plt.show()

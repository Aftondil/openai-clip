import os
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader

# Updated class names to match your actual folder structure
classnames = [
    "a photo of fresh fruit",
    "a photo of dried fruit",
    "a photo of vegetables like potatoes and onion",
    "a photo of rice and sunflower seed",
    "a photo of spices and medicinal herb",
    "a photo of egg",
    "a photo of flowers and seedling",
    "a photo of bread",
    "a photo of plastic and paper bag",
    "a photo of closed market stall"
]

# Create label dictionary
label_dict = {name: i for i, name in enumerate(classnames)}


class CustomImageDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.transform = transform

        # Initialize lists to store image paths and labels
        self.image_paths = []
        self.labels = []

        print(f"Looking for classes in: {folder_path}")

        # Loop through class folders
        for label_name in classnames:
            label_path = os.path.join(self.folder_path, label_name)
            print(f"Checking folder: {label_path}")

            if os.path.exists(label_path):
                image_files = [f for f in os.listdir(label_path)
                               if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
                print(f"Found {len(image_files)} images in {label_name}")

                for filename in image_files:
                    image_path = os.path.join(label_path, filename)
                    self.image_paths.append(image_path)
                    self.labels.append(label_dict[label_name])
            else:
                print(f"Warning: Folder does not exist: {label_path}")

        print(f"Total images loaded: {len(self.image_paths)}")
        if len(self.labels) > 0:
            print(f"Label range: {min(self.labels)} to {max(self.labels)}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')

        if self.transform is not None:
            image = self.transform(image)

        label = self.labels[idx]
        return image, label


# Main dataset class that WiSE-FT expects
class YourCustomDataset:
    def __init__(self, preprocess,
                 location=os.path.expanduser('~/your_dataset'),
                 batch_size=128,
                 num_workers=16,
                 classnames=None):

        print(f"Initializing dataset from location: {location}")

        # CRITICAL: Always use our custom classnames, never the passed ones
        # This ensures consistency between zero-shot and fine-tuning phases
        self.classnames = globals()['classnames']
        print(f"Using {len(self.classnames)} classes: {self.classnames}")

        # Training dataset
        train_path = os.path.join(location, 'train')
        print(f"Train path: {train_path}")

        self.train_dataset = CustomImageDataset(
            folder_path=train_path,
            transform=preprocess
        )

        if len(self.train_dataset) == 0:
            raise ValueError(f"No training images found in {train_path}. Please check your dataset structure.")

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers
        )

        # Test dataset (if you have separate test data)
        test_path = os.path.join(location, 'test')
        if os.path.exists(test_path):
            self.test_dataset = CustomImageDataset(
                folder_path=test_path,
                transform=preprocess
            )
        else:
            # Use train data for testing if no separate test set
            self.test_dataset = self.train_dataset

        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )


# Text templates for your domain - CRITICAL: These must match your classnames exactly
your_custom_template = [
    lambda c: f"{c}.",  # Direct use since your classnames are already full descriptions
    lambda c: f"this is {c}.",
    lambda c: f"an image of {c}.",
    lambda c: f"a picture showing {c}.",
    lambda c: f"a photograph of {c}.",
]
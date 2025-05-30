import os
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader

# Define your class names
classnames = [
    "a photo of fresh fruits on a market stall",
    "a photo of citrus and tropical fruits on a market stall",
    "a photo of dried fruits on a market stall",
    "a photo of vegetables like potatoes and onions on a market stall",
    "a photo of tomatoes and cucumbers sold in winter on a market stall",
    "a photo of rice and sunflower seeds on a market stall",
    "a photo of legumes like mung beans on a market stall",
    "a photo of dairy products like milk on a market stall",
    "a photo of pickled foods on a market stall",
    "a photo of spices and medicinal herbs on a market stall",
    "a photo of fresh herbs on a market stall",
    "a photo of eggs on a market stall",
    "a photo of flowers and seedlings on a market stall",
    "a photo of bread on a market stall",
    "a photo of homemade preserved foods on a market stall",
    "a photo of industrial packaged food products on a market stall",
    "a photo of plastic and paper bags on a market stall",
    "a photo of brooms on a market stall",
    # "a photo of live fish on a market stall",
    "a photo of meat products on a market stall",
    # "a photo of poultry meat and offal on a market stall",
    # "a photo of products sold from light vehicles at a market",
    # "a photo of products sold from cargo trucks at a market",
    "a photo of products sold on the ground at a market",
    "a photo of a closed or inactive market stall"
] # Replace with your actual class names

# Create label dictionary
label_dict = {name: i for i, name in enumerate(classnames)}


class CustomImageDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.transform = transform

        # Initialize lists to store image paths and labels
        self.image_paths = []
        self.labels = []

        # Loop through class folders
        for label_name in classnames:
            label_path = os.path.join(self.folder_path, label_name)
            if os.path.exists(label_path):
                for filename in os.listdir(label_path):
                    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                        image_path = os.path.join(label_path, filename)
                        self.image_paths.append(image_path)
                        self.labels.append(label_dict[label_name])

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

        # Training dataset
        self.train_dataset = CustomImageDataset(
            folder_path=os.path.join(location, 'train'),
            transform=preprocess
        )
        self.train_dataloader = DataLoader(
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

        self.test_dataloader = DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )

        self.classnames = classnames if classnames else globals()['classnames']


# Text templates for your domain (customize these for your specific classes)
your_custom_template = [
    lambda c: f"a photo of a {c}.",
    lambda c: f"this is a {c}.",
    lambda c: f"a {c} image.",
    lambda c: f"an image of {c}.",
    lambda c: f"the {c} is shown.",
    lambda c: f"a picture of {c}.",
    lambda c: f"{c} photo.",
]
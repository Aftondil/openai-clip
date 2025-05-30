import os
import json
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader


class JSONImageDataset(Dataset):
    def __init__(self, json_path, root_dir, transform=None):
        """
        Args:
            json_path: Path to the JSON file containing image paths and labels
            root_dir: Root directory containing the images
            transform: Optional transform to be applied on images
        """
        self.root_dir = root_dir
        self.transform = transform

        # Load JSON data (handle both JSON and JSONL formats)
        self.data = []
        with open(json_path, 'r') as f:
            # Try to load as regular JSON first
            try:
                f.seek(0)
                content = f.read().strip()
                if content.startswith('['):
                    # It's a JSON array
                    self.data = json.loads(content)
                else:
                    # It's JSONL format (one JSON object per line)
                    f.seek(0)
                    for line in f:
                        line = line.strip()
                        if line:
                            self.data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"JSON decode error: {e}")
                # Try JSONL format as fallback
                f.seek(0)
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if line:
                        try:
                            self.data.append(json.loads(line))
                        except json.JSONDecodeError as line_error:
                            print(f"Error parsing line {line_num}: {line_error}")
                            print(f"Problematic line: {line[:100]}...")
                            continue

        # Extract unique class labels to create mapping
        self.class_labels = sorted(list(set(item['class_label'] for item in self.data)))
        self.label_to_idx = {label: idx for idx, label in enumerate(self.class_labels)}

        print(f"Found {len(self.class_labels)} classes: {self.class_labels}")
        print(f"Loaded {len(self.data)} samples from {json_path}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Get image path (relative to root_dir)
        image_path = os.path.join(self.root_dir, item['image_path'])

        # Load and convert image
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return a black image as fallback
            image = Image.new('RGB', (224, 224), color='black')

        # Apply transforms
        if self.transform is not None:
            image = self.transform(image)

        # Get label index
        label = self.label_to_idx[item['class_label']]

        return image, label


# Main dataset class that WiSE-FT expects
class ProductDataset:
    def __init__(self, preprocess,
                 location=os.path.expanduser('~/your_dataset'),
                 batch_size=128,
                 num_workers=16,
                 classnames=None):

        # Define paths
        train_json = os.path.join(location, 'train_data.json')
        val_json = os.path.join(location, 'val_data.json')
        test_json = os.path.join(location, 'test_data.json')

        # Training dataset
        self.train_dataset = JSONImageDataset(
            json_path=train_json,
            root_dir=location,
            transform=preprocess
        )

        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers
        )

        # Create aliases for compatibility with WiSE-FT code
        self.train_loader = self.train_dataloader

        # Validation/Test dataset
        if os.path.exists(test_json):
            self.test_dataset = JSONImageDataset(
                json_path=test_json,
                root_dir=location,
                transform=preprocess
            )
        elif os.path.exists(val_json):
            self.test_dataset = JSONImageDataset(
                json_path=val_json,
                root_dir=location,
                transform=preprocess
            )
        else:
            print("No test or validation JSON found, using train data for evaluation")
            self.test_dataset = self.train_dataset

        self.test_dataloader = DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )

        # Create aliases for compatibility with WiSE-FT code
        self.test_loader = self.test_dataloader

        # Use class names from the dataset
        self.classnames = self.train_dataset.class_labels
        print(f"Dataset classnames: {self.classnames}")


# Text templates for product/fashion domain
# Customize these based on your specific product types
product_template = [
    lambda c: f"a photo of a {c}.",
    lambda c: f"this is a {c}.",
    lambda c: f"a {c} product.",
    lambda c: f"an image of a {c}.",
    lambda c: f"a {c} item.",
    lambda c: f"a picture of a {c}.",
    lambda c: f"this {c} is shown.",
    lambda c: f"a {c} for sale.",
    lambda c: f"product image of a {c}.",
    lambda c: f"a {c} in the image.",
]

# Alternative: More specific templates for fashion/clothing items
fashion_template = [
    lambda c: f"a photo of a {c}.",
    lambda c: f"this is a {c}.",
    lambda c: f"a beautiful {c}.",
    lambda c: f"an elegant {c}.",
    lambda c: f"a stylish {c}.",
    lambda c: f"a fashionable {c}.",
    lambda c: f"a {c} garment.",
    lambda c: f"traditional {c}.",
    lambda c: f"handwoven {c}.",
    lambda c: f"ethnic {c}.",
]


# You can also create class-specific templates if needed
def create_class_specific_templates():
    """
    Create templates that are more specific to certain product types
    """
    templates = {
        'saree': [
            lambda c: f"a beautiful {c}.",
            lambda c: f"an elegant {c}.",
            lambda c: f"a traditional {c}.",
            lambda c: f"a handwoven {c}.",
            lambda c: f"an ethnic {c}.",
        ],
        # Add more product-specific templates as needed
        'default': product_template
    }
    return templates
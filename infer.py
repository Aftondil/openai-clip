import torch
from PIL import Image
import torchvision.transforms as transforms
from src.models.modeling import ImageClassifier


def create_wiseft_model(zeroshot_checkpoint, finetuned_checkpoint, alpha=0.5):
    """
    Create WiSE-FT model with weight interpolation
    alpha=0: pure zero-shot, alpha=1: pure fine-tuned, alpha=0.5: balanced
    """
    # Load models
    zeroshot = ImageClassifier.load(zeroshot_checkpoint)
    finetuned = ImageClassifier.load(finetuned_checkpoint)

    # Get state dicts
    theta_0 = zeroshot.state_dict()
    theta_1 = finetuned.state_dict()

    # Make sure checkpoints are compatible
    assert set(theta_0.keys()) == set(theta_1.keys())

    # Interpolate between checkpoints with mixing coefficient alpha
    theta = {
        key: (1 - alpha) * theta_0[key] + alpha * theta_1[key]
        for key in theta_0.keys()
    }

    # Update the model according to the new weights
    finetuned.load_state_dict(theta)
    finetuned.eval()

    return finetuned


# Usage
zeroshot_checkpoint = "./models/wiseft/bozor-v2/zeroshot.pt"
finetuned_checkpoint = "./models/wiseft/bozor-v2/finetuned/checkpoint_70.pt"

# Create model with different interpolation values
# model_balanced = create_wiseft_model(zeroshot_checkpoint, finetuned_checkpoint, alpha=0.5)
# model_conservative = create_wiseft_model(zeroshot_checkpoint, finetuned_checkpoint, alpha=0.3)
# model_aggressive = create_wiseft_model(zeroshot_checkpoint, finetuned_checkpoint, alpha=0.8)
model = ImageClassifier.load("models/wiseft/bozor-v2/finetuned/wise_ft_alpha=0.700.pt")
# Classify with the balanced model
class_names = [
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
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    # transforms.CenterCrop(224),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

image = Image.open("images-test/camera_67_roi_15776.png").convert('RGB')
image_tensor = transform(image).unsqueeze(0)

with torch.no_grad():
    logits = model(image_tensor)
    probabilities = torch.softmax(logits, dim=1)
    predicted_class = torch.argmax(logits, dim=1).item()

print(f"Predicted: {class_names[predicted_class]} ({probabilities[0][predicted_class]:.4f})")
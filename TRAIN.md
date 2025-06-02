# Custom Dataset Classes for WiSE-FT

This repository contains two custom dataset implementations for the WiSE-FT (Robust fine-tuning of zero-shot models) framework, allowing you to fine-tune CLIP models on your own datasets while maintaining out-of-distribution robustness.

## Dataset Classes

### 1. Folder-Based Dataset (`CustomImageDataset`)
For datasets organized in traditional folder structure where each class has its own folder.

### 2. JSON-Based Dataset (`ProductDataset`) 
For datasets with metadata stored in JSON/JSONL files, particularly useful for e-commerce, fashion, or any domain with rich metadata.

## Installation & Setup

```bash
# Clone the WiSE-FT repository
git clone https://github.com/mlfoundations/wise-ft.git
cd wise-ft

# Create and activate conda environment
conda env create
conda activate wiseft

# Set Python path
export PYTHONPATH="$PYTHONPATH:$PWD"

# Copy the custom dataset file to the datasets directory
cp product_dataset.py src/datasets/
```

## Dataset Format Requirements

### Folder-Based Dataset Structure
```
your_dataset/
├── train/
│   ├── class1/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   ├── class2/
│   │   ├── image1.jpg
│   │   └── ...
│   └── class3/
│       └── ...
└── test/ (optional)
    ├── class1/
    ├── class2/
    └── class3/
```

### JSON-Based Dataset Structure
```
your_dataset/
├── images/
│   ├── train/
│   ├── test/
│   └── val/
├── train_data.json
├── test_data.json
└── val_data.json
```

**JSON Format** (supports both JSON array and JSONL formats):
```json
{"image_url": "https://example.com/image.jpg", "image_path": "images/train/0.jpeg", "brand": "Brand Name", "product_title": "Product Description", "class_label": "category_name"}
{"image_url": "https://example.com/image2.jpg", "image_path": "images/train/1.jpeg", "brand": "Another Brand", "product_title": "Another Product", "class_label": "another_category"}
```

## Configuration

### 1. Register Your Dataset

Add your dataset to `src/datasets/__init__.py`:

```python
# For folder-based dataset
from .your_dataset import YourCustomDataset, your_custom_template

# For JSON-based dataset  
from .product_dataset import ProductDataset, fashion_template, product_template
```

### 2. Customize Text Templates

**Fashion/Clothing Templates:**
```python
fashion_template = [
    lambda c: f"a photo of a {c}.",
    lambda c: f"this is a {c}.",
    lambda c: f"a beautiful {c}.",
    lambda c: f"an elegant {c}.",
    lambda c: f"a stylish {c}.",
    lambda c: f"a fashionable {c}.",
    lambda c: f"traditional {c}.",
    lambda c: f"handwoven {c}.",
]
```

**General Product Templates:**
```python
product_template = [
    lambda c: f"a photo of a {c}.",
    lambda c: f"this is a {c}.",
    lambda c: f"a {c} product.",
    lambda c: f"an image of a {c}.",
    lambda c: f"a {c} item.",
    lambda c: f"product image of a {c}.",
]
```

## Fine-tuning Commands

### JSON-Based Dataset (Fashion/E-commerce)

```bash
python -m src.wise_ft \
    --train-dataset=ProductDataset \
    --epochs=10 \
    --lr=0.00003 \
    --batch-size=64 \
    --model=ViT-B/32 \
    --eval-datasets=ProductDataset \
    --template=fashion_template \
    --results-db=results.jsonl \
    --save=models/wiseft/fashion_model \
    --data-location=./path/to/your/dataset \
    --alpha 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
```

### Folder-Based Dataset

```bash
python -m src.wise_ft \
    --train-dataset=YourCustomDataset \
    --epochs=10 \
    --lr=0.00003 \
    --batch-size=256 \
    --model=ViT-B/32 \
    --eval-datasets=YourCustomDataset \
    --template=your_custom_template \
    --results-db=results.jsonl \
    --save=models/wiseft/your_model \
    --data-location=./dataset \
    --alpha 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
```

## Key Parameters

| Parameter | Description | Example Values |
|-----------|-------------|----------------|
| `--train-dataset` | Name of your dataset class | `ProductDataset`, `YourCustomDataset` |
| `--model` | CLIP model variant | `ViT-B/32`, `ViT-B/16`, `ViT-L/14` |
| `--template` | Text template set | `fashion_template`, `product_template` |
| `--epochs` | Training epochs | `5-20` (start with 10) |
| `--lr` | Learning rate | `0.00001-0.0001` |
| `--batch-size` | Batch size | `32-128` (adjust based on GPU memory) |
| `--alpha` | Weight interpolation values | `0 0.1 0.2 ... 1.0` |
| `--freeze-encoder` | Only train classifier head | Add flag to freeze encoder |

## Understanding Alpha Values

The `--alpha` parameter controls the interpolation between zero-shot and fine-tuned weights:

- **α = 0.0**: Pure zero-shot model (original CLIP)
- **α = 0.5**: 50% zero-shot + 50% fine-tuned  
- **α = 1.0**: Pure fine-tuned model
- **Optimal α**: Usually between 0.1-0.7, found through evaluation

## Training Process

1. **Zero-shot Evaluation**: Tests original CLIP performance on your dataset
2. **Fine-tuning**: Trains CLIP on your specific dataset  
3. **Weight Interpolation**: Creates models with different α values
4. **Evaluation**: Tests all α values to find optimal balance
5. **Results**: Saves best performing model and generates performance plots

## Expected Output

```
Getting zeroshot weights.
Saving image classifier to models/wiseft/your_model/zeroshot.pt
Fine-tuning end-to-end
Found X classes: ['class1', 'class2', ...]
Loaded Y samples from ./dataset/train_data.json
Training... [Progress bars]
Evaluating alpha values...
Best performance at alpha=0.3
```

## Monitoring Training

The training generates several outputs:

- **Models**: Saved in `models/wiseft/your_model/`
- **Results**: Performance metrics in `results.jsonl`
- **Plots**: Accuracy vs. robustness scatter plots
- **Logs**: Training progress and evaluation results

## Troubleshooting

### Common Issues

**1. Module Import Errors**
```bash
# Ensure you're in the wise-ft directory and PYTHONPATH is set
export PYTHONPATH="$PYTHONPATH:$PWD"
```

**2. JSON Format Errors**
- The dataset class handles both JSON arrays and JSONL formats
- Ensure image paths are relative to the dataset root directory

**3. Memory Issues**
- Reduce `--batch-size` (try 32 or 16)
- Reduce `--num-workers` in dataset class

**4. Missing Images**
- Check that `image_path` in JSON files is correct
- Verify images exist at specified paths

### Performance Tips

**1. Start Small**
- Test with a subset of data first
- Use smaller batch sizes initially
- Try fewer epochs (5-10) for initial experiments

**2. Template Optimization**  
- Create domain-specific templates
- Test different template sets
- Consider class-specific templates for better performance

**3. Hyperparameter Tuning**
- Learning rate: Start with `3e-5`, adjust based on convergence
- Batch size: Balance between speed and memory usage
- Epochs: Monitor validation loss to avoid overfitting

## Advanced Usage

### Custom Templates for Specific Domains

```python
# Medical imaging
medical_template = [
    lambda c: f"a medical image showing {c}.",
    lambda c: f"this is a {c} scan.",
    lambda c: f"radiological image of {c}.",
]

# Satellite imagery  
satellite_template = [
    lambda c: f"satellite image of {c}.",
    lambda c: f"aerial view of {c}.",
    lambda c: f"remote sensing image showing {c}.",
]
```

### Class-Specific Templates

```python
def get_class_specific_template(class_name):
    templates = {
        'saree': [lambda c: f"traditional Indian {c}.", lambda c: f"elegant {c}."],
        'kurta': [lambda c: f"ethnic {c}.", lambda c: f"traditional {c}."],
        'default': fashion_template
    }
    return templates.get(class_name, templates['default'])
```

## Results Analysis

After training, analyze the results:

1. **Check `results.jsonl`** for detailed metrics
2. **Review scatter plots** showing accuracy vs. robustness trade-offs  
3. **Compare different α values** to understand the interpolation effect
4. **Evaluate on out-of-distribution test sets** to verify robustness

## Citation

If you use this code, please cite the original WiSE-FT paper:

```bibtex
@article{wortsman2021robust,
  title={Robust fine-tuning of zero-shot models},
  author={Wortsman, Mitchell and Ilharco, Gabriel and Kim, Jong Wook and Li, Mike and Kornblith, Simon and Roelofs, Rebecca and Gontijo-Lopes, Raphael and Hajishirzi, Hannaneh and Farhadi, Ali and Namkoong, Hongseok and Schmidt, Ludwig},
  journal={arXiv preprint arXiv:2109.01903},
  year={2021}
}
```

## Support

For issues specific to the custom dataset implementations, please check:

1. Dataset format and structure
2. JSON file validity  
3. Image path correctness
4. Template customization for your domain

For general WiSE-FT issues, refer to the [original repository](https://github.com/mlfoundations/wise-ft).
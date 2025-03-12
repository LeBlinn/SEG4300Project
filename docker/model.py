import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import random
import numpy as np

# Set seed for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior in CUDA (if using a GPU)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  

set_seed(42)  # Call this before loading the model

# Define the number of output classes (update this based on your dataset)
NUM_CLASSES = 5  

# Pretrained models dictionary
PRETRAINED_MODELS = {
    'resnet18': {
        'model': models.resnet18,
        'weights': models.ResNet18_Weights.IMAGENET1K_V1,
        'feature_dim': 512,
        'classifier_layer': 'fc'
    }
}

# Define the model class
class BatteryClassifier(nn.Module):
    def __init__(self, model_name, num_classes=NUM_CLASSES, dropout_rate=0.5):
        super().__init__()
        self.model_config = PRETRAINED_MODELS[model_name]

        # Load pretrained model
        self.base_model = self.model_config['model'](weights=self.model_config['weights'])

        # Freeze all layers initially
        for param in self.base_model.parameters():
            param.requires_grad = False

        # Unfreeze classifier/decoder layers
        self._unfreeze_decoder_layers(model_name)

        # Replace classifier
        self._build_classifier(num_classes, dropout_rate, model_name)

    def _unfreeze_decoder_layers(self, model_name):
        if 'resnet' in model_name:
            for param in self.base_model.layer3.parameters():
                param.requires_grad = True
            for param in self.base_model.layer4.parameters():
                param.requires_grad = True

    def _build_classifier(self, num_classes, dropout_rate, model_name):
        in_features = self.model_config['feature_dim']

        classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, 512),
            nn.GELU(),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout_rate/2),
            nn.Linear(512, num_classes)
        )

        # Replace the classifier layer
        setattr(self.base_model, self.model_config['classifier_layer'], classifier)

    def forward(self, x):
        return self.base_model(x)

# Model name (should match what was used during training)
MODEL_NAME = "resnet18"

# Load model and weights
model = BatteryClassifier(model_name=MODEL_NAME)
model.load_state_dict(torch.load("modelv1.pth", map_location=torch.device("cpu")))
model.eval()

# Define image preprocessing steps
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # ResNet expects 224x224 images
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
])

def load_and_predict(image_path):
    """Loads an image, preprocesses it, and runs inference."""
    image = Image.open(image_path).convert("RGB")  
    input_tensor = transform(image).unsqueeze(0)  # Apply transforms & add batch dimension

    # Run inference
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = nn.Softmax(dim=1)(output)  # Apply softmax to get class probabilities
        confidences, predicted_class = probabilities.max(1)  # Get the class with highest probability and its confidence score

    return predicted_class.item(), confidences.item()  # Return both predicted class and confidence score
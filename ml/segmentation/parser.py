import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from models.bisenet import BiSeNet

class FaceParser:

    def __init__(
        self,
        model_path,
        image_size=512
    ):

        self.image_size = image_size

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        # Create BiSeNet model
        self.model = BiSeNet(
            num_classes=19,
            backbone_name="resnet18"
        )

        # Load trained weights
        self.model.load_state_dict(
            torch.load(
                model_path,
                map_location=self.device
            )
        )

        self.model.to(self.device)

        self.model.eval()

        # ImageNet normalization
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)
            )
        ])

    @torch.no_grad()
    def predict(self, image):

        # Original dimensions
        original_height, original_width = image.shape[:2]

        # OpenCV BGR → RGB
        rgb = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        # Convert NumPy → PIL
        pil_image = Image.fromarray(rgb)

        # Resize
        pil_image = pil_image.resize(
            (
                self.image_size,
                self.image_size
            ),
            Image.BILINEAR
        )

        # Normalize + convert to tensor
        tensor = self.transform(
            pil_image
        )

        # Add batch dimension
        tensor = tensor.unsqueeze(0)

        # Move to CPU/GPU
        tensor = tensor.to(self.device)

        # Model inference
        output = self.model(tensor)[0]

        # Get class with highest probability
        prediction = torch.argmax(
            output,
            dim=1
        )

        # Remove batch dimension
        prediction = (
            prediction
            .squeeze(0)
            .cpu()
            .numpy()
        )

        # Resize segmentation mask
        # back to original image dimensions
        prediction = cv2.resize(
            prediction.astype(np.uint8),
            (
                original_width,
                original_height
            ),
            interpolation=cv2.INTER_NEAREST
        )

        return prediction
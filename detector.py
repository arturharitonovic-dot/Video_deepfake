import cv2
import torch
from torchvision import transforms, models
from facenet_pytorch import MTCNN
from PIL import Image


class DeepfakeDetector:
    def __init__(self, model_weights_path: str = None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.mtcnn = MTCNN(keep_all=False, device=self.device, post_process=False)

        self.model = models.resnet18(weights=None)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 1)

        if model_weights_path:
            self.model.load_state_dict(torch.load(model_weights_path, map_location=self.device))

        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def process_frame(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_frame)

        boxes, _ = self.mtcnn.detect(pil_img)

        if boxes is None or len(boxes) == 0:
            return None, None

        box = boxes[0].astype(int)
        x1, y1, x2, y2 = max(0, box[0]), max(0, box[1]), box[2], box[3]

        width, height = pil_img.size
        x1, y1 = min(x1, width), min(y1, height)
        x2, y2 = min(x2, width), min(y2, height)

        if x2 <= x1 or y2 <= y1:
            return None, None

        face_crop = pil_img.crop((x1, y1, x2, y2)).resize((224, 224))

        input_tensor = self.transform(face_crop).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(input_tensor)
            probability = torch.sigmoid(output).item()

        return probability, (x1, y1, x2, y2)
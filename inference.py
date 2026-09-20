import torch
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import Resize
from transformers import AutoImageProcessor, ResNetForImageClassification


MODEL_NAME = "microsoft/resnet-18"
BATCH_SIZE = 64

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)

processor = AutoImageProcessor.from_pretrained(MODEL_NAME)

dataset = MNIST(
    root="./data",
    train=False,
    download=True,
)

resize = Resize((224, 224))

def collate_fn(batch):
    images, labels = zip(*batch)
    images = [resize(image.convert("RGB")) for image in images]
    inputs = processor(images=images, do_resize=False, return_tensors="pt")
    return inputs["pixel_values"], torch.tensor(labels)


loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn,
)

model = ResNetForImageClassification.from_pretrained(
    MODEL_NAME,
    num_labels=10,
    ignore_mismatched_sizes=True,
).to(device)

model.eval()

correct = 0
total = 0

with torch.no_grad():
    for pixel_values, labels in loader:
        pixel_values = pixel_values.to(device)
        labels = labels.to(device)

        logits = model(pixel_values=pixel_values).logits
        predictions = logits.argmax(dim=-1)

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

accuracy = correct / total
print(f"Accuracy: {accuracy:.4f}")

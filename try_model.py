import torch
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images

device = "cpu"
dtype = torch.float32

# init empty model (no weights)
model = VGGT()
print("Model initialized without pretrained weights")

# load one small image
image_names = ["path/to/one_tiny_image.png"]
images = load_and_preprocess_images(image_names).to(device)

with torch.no_grad():
    _ = model.encoder(images)   # sanity check only
print("Forward pass ok")

from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import os

class FaceOcclusionDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.images = os.listdir(data_dir)
        self.transform = transform or transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_dir, self.images[idx])
        img = Image.open(img_path).convert("RGB")
        label = self.images[idx].split("_")[0]  # occlusion type
        identity = int(self.images[idx].split("_")[-1].split(".")[0])  # assumed to be last
        return self.transform(img), label, identity

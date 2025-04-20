import clip
import torch

class CLIPIdentifier:
    def __init__(self, device="cuda"):
        self.model, _ = clip.load("ViT-B/32", device=device)
        self.device = device
        self.occlusion_types = ["mask", "sunglasses", "hat"]
        self.embeddings = self._generate_embeddings()

    def _generate_embeddings(self):
        texts = clip.tokenize(self.occlusion_types).to(self.device)
        with torch.no_grad():
            return self.model.encode_text(texts)

    def get_embedding(self, label):
        idx = self.occlusion_types.index(label)
        return self.embeddings[idx]

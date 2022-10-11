from transformers import CLIPTokenizer, CLIPTextModel, CLIPVisionModel, CLIPFeatureExtractor
import torch
import PIL

class AbstractEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError

class FrozenCLIPTextEmbedder(AbstractEncoder):
    """Uses the CLIP transformer encoder for text (from Hugging Face)"""
    def __init__(self, version="openai/clip-vit-large-patch14", device="cuda", max_length=77):
        super().__init__()
        self.tokenizer = CLIPTokenizer.from_pretrained(version)
        self.transformer = CLIPTextModel.from_pretrained(version)
        self.device = device
        self.max_length = max_length
        self.freeze()

    def freeze(self):
        self.transformer = self.transformer.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        batch_encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        tokens = batch_encoding["input_ids"].to(self.device)
        outputs = self.transformer(input_ids=tokens)

        z = outputs.last_hidden_state
        return z

    def encode(self, text):
        return self(text)

class FrozenCLIPVisionEmbedder(AbstractEncoder):
    """Uses the CLIP transformer encoder for images (from Hugging Face)"""
    def __init__(self, version: str = "openai/clip-vit-large-patch14", device: str = "cuda"):
        super().__init__()
        self.featextractor = CLIPFeatureExtractor.from_pretrained(version)
        self.transformer = CLIPVisionModel.from_pretrained(version)
        self.device = device
        self.freeze()

    def freeze(self):
        self.transformer = self.transformer.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, image: PIL.Image):
        batch_encoding = self.featextractor(images = image, return_tensors = 'pt')
        pixel_values = batch_encoding['pixel_values'].to(self.device)
        outputs = self.transformer(pixel_values = pixel_values)
        z = outputs.last_hidden_state
        return z

    def encode(self, image: PIL.Image):
        return self(image)

def dbg_embedding_info(emb: torch.Tensor = None):
    assert emb is not None
    print(f'emb_info:\n\tbs: {len(emb)}\n\tlen: {len(emb[-1])}\n\tfeat_len: {len(emb[-1][-1])}')
    return emb

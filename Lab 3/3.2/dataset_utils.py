from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder

# Mappatura tra ID di classe di ImageNette
IMAGENETTE_CLASSES = {
    'n01440764': 'tench',
    'n02102040': 'English springer',
    'n02979186': 'cassette player',
    'n03000684': 'chain saw',
    'n03028079': 'church',
    'n03394916': 'French horn',
    'n03417042': 'garbage truck',
    'n03425413': 'gas pump',
    'n03445777': 'golf ball',
    'n03888257': 'parachute'
}

# Dataset personalizzato per il modello CLIP
class CLIPImageDataset(Dataset):
    def __init__(self, image_folder, processor, class_names):
        self.dataset = ImageFolder(image_folder)
        self.processor = processor
        self.class_names = class_names
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        # Process image
        inputs = self.processor(images=image, return_tensors="pt", padding=True)
        # Remove batch dimension
        pixel_values = inputs['pixel_values'].squeeze(0)
        
        return {
            'pixel_values': pixel_values,
            'labels': label
        }

# Funzione per creare text prompts
def create_text_prompts(class_names, template="a photo of a {}"):
    return [template.format(name) for name in class_names]
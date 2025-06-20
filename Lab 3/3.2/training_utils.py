
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import accuracy_score

# Valutazione zero-shot del modello CLIP 
def evaluate_zero_shot(model, processor, dataloader, text_prompts, device):
    model.eval()
    
    # Codifica le frasi testuali una sola volta (sono statiche)
    text_inputs = processor(text=text_prompts, return_tensors="pt", padding=True)
    text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
    
    with torch.no_grad():
        text_features = model.get_text_features(**text_inputs)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Zero-shot evaluation"):
            pixel_values = batch['pixel_values'].to(device)
            labels = batch['labels'].to(device)
            
            # Estrae le feature dell'immagine
            image_features = model.get_image_features(pixel_values=pixel_values)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            # Calcola la similarità tra immagine e testo (via prodotto scalare)
            logits = (image_features @ text_features.T) * model.logit_scale.exp()
            predictions = logits.argmax(dim=-1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_predictions)
    return accuracy, all_predictions, all_labels

# Training function con PEFT
def train_epoch(model, processor, dataloader, text_prompts, optimizer, device):
    
    model.train()
    total_loss = 0
    
    # Codifica i prompt testuali 
    text_inputs = processor(text=text_prompts, return_tensors="pt", padding=True)
    text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
    
    for batch in tqdm(dataloader, desc="Training"):
        pixel_values = batch['pixel_values'].to(device)
        labels = batch['labels'].to(device)
        
        # Forward pass
        image_features = model.get_image_features(pixel_values=pixel_values)
        text_features = model.get_text_features(**text_inputs)
        
        # Normalizza le feature vettoriali
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # Calcola la similarità immagine-testo moltiplicata per logit scale
        logits = (image_features @ text_features.T) * model.logit_scale.exp()
        
        # Calcola loss (cross-entropy)
        loss = nn.CrossEntropyLoss()(logits, labels)
        
        # Backpropagation e aggiornamento dei pesi LoRA
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)
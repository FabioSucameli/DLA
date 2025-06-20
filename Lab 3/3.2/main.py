import os
import torch
import torch.optim as optim
import numpy as np
import wandb
from torch.utils.data import DataLoader
from transformers import CLIPProcessor, CLIPModel
from peft import LoraConfig, get_peft_model, TaskType
from pathlib import Path
from sklearn.metrics import classification_report

from dataset_utils import IMAGENETTE_CLASSES, CLIPImageDataset, create_text_prompts
from training_utils import evaluate_zero_shot, train_epoch


os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Configurazione
config = {
    "model_name": "openai/clip-vit-base-patch16",
    "batch_size": 32,
    "learning_rate": 1e-4,
    "num_epochs": 10,
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.1,
    "seed": 42,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "text_template": "a photo of a {}", 
}

# Imposta i seed per la riproducibilità
torch.manual_seed(config["seed"])
np.random.seed(config["seed"])

# Inizializza wandb
wandb.init(
    project="clip-imagenette-finetuning",
    config=config,
    name="clip-peft-experiment"
)

# Main
def main():
    # Percorso dataset
    imagenette_path =  Path("./data/imagenette2")
    
    # Carica il modello CLIP pre-addestrato e il suo processor
    print(f"Loading CLIP model: {config['model_name']}")
    model = CLIPModel.from_pretrained(config['model_name'])
    processor = CLIPProcessor.from_pretrained(config['model_name'])
    
    # Sposta il modello su GPU (se disponibile)
    model = model.to(config['device'])
    
    # Prepara i percorsi per il training e la validazione
    train_path = imagenette_path / "train"
    val_path = imagenette_path / "val"
    
    # Ottiene i nomi delle classi in ordine coerente
    class_folders = sorted(os.listdir(train_path))
    class_names = [IMAGENETTE_CLASSES[folder] for folder in class_folders]
    
    # Crea i dataset per il training e la validazione
    train_dataset = CLIPImageDataset(train_path, processor, class_names)
    val_dataset = CLIPImageDataset(val_path, processor, class_names)
    
    # Crea i dataloader PyTorch per batching
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Crea text prompts
    text_prompts = create_text_prompts(class_names)
    
    # Valutazione Zero-shot
    print("\n=== Zero-shot Evaluation ===")
    zero_shot_acc, _, _ = evaluate_zero_shot(
        model, processor, val_loader, text_prompts, config['device']
    )
    print(f"Zero-shot accuracy: {zero_shot_acc:.4f}")
    wandb.log({"zero_shot_accuracy": zero_shot_acc})
    
    # Configura LoRA per essere applicato sia all'image encoder che al text encoder
    print("\n=== Configuring LoRA ===")
    
    #Configurazione di LoRA
    peft_config = LoraConfig(
        r=config['lora_r'],
        lora_alpha=config['lora_alpha'],
        target_modules=["k_proj", "v_proj", "q_proj", "out_proj"],
        lora_dropout=config['lora_dropout'],
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION,
    )
    
    # Applica LoRA al modello completo
    model = get_peft_model(model, peft_config)
    
    # Stampa quanti parametri sono aggiornabili rispetto al totale
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"Trainable params: {trainable_params:,}")
    print(f"Total params: {total_params:,}")
    print(f"Trainable ratio: {trainable_params / total_params:.2%}")
    
    # Inizializza l’ottimizzatore AdamW con il learning rate configurato
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'])
    
    # Training loop
    print("\n=== Fine-tuning with LoRA ===")
    best_val_acc = 0
    
    for epoch in range(config['num_epochs']):
        print(f"\nEpoch {epoch + 1}/{config['num_epochs']}")
        
        # Train
        train_loss = train_epoch(
            model, processor, train_loader, text_prompts, optimizer, config['device']
        )
        
        # Valutazione
        val_acc, val_preds, val_labels = evaluate_zero_shot(
            model, processor, val_loader, text_prompts, config['device']
        )
        
        # Log metrics
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_accuracy": val_acc,
            "learning_rate": optimizer.param_groups[0]['lr']
        })
        
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Accuracy: {val_acc:.4f}")
        
        # Salva il modello
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_accuracy': val_acc,
            }, 'best_clip_lora_model.pth')
            print(f"New best model saved! Accuracy: {val_acc:.4f}")
    
    # Valutazione finale dopo tutte le epoche
    print("\n=== Final Results ===")
    print(f"Zero-shot accuracy: {zero_shot_acc:.4f}")
    print(f"Best fine-tuned accuracy: {best_val_acc:.4f}")
    print(f"Improvement: {(best_val_acc - zero_shot_acc):.4f} ({(best_val_acc - zero_shot_acc) / zero_shot_acc * 100:.1f}%)")
    
    # Log finali
    wandb.log({
        "final_zero_shot_acc": zero_shot_acc,
        "final_best_acc": best_val_acc,
        "improvement": best_val_acc - zero_shot_acc
    })
    
    # Genera un report di classificazione dettagliato
    _, final_preds, final_labels = evaluate_zero_shot(
        model, processor, val_loader, text_prompts, config['device']
    )
    
    report = classification_report(
        final_labels, 
        final_preds, 
        target_names=class_names,
        output_dict=True
    )
    
    # Registra il report di classificazione su wandb come tabella
    wandb.log({"classification_report": wandb.Table(
        columns=["class", "precision", "recall", "f1-score", "support"],
        data=[[cls, metrics['precision'], metrics['recall'], 
               metrics['f1-score'], metrics['support']] 
              for cls, metrics in report.items() if cls not in ['accuracy', 'macro avg', 'weighted avg']]
    )})
    
    wandb.finish()

if __name__ == "__main__":
    main()

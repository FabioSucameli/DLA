import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import Subset, DataLoader
from tqdm import tqdm
from sklearn.metrics import classification_report
from torch.utils.tensorboard import SummaryWriter

# Modello MLP
class MLP(nn.Module):
    def __init__(self, input_size=784, hidden_sizes=[16, 16], output_size=10, name="MLP"):
        super().__init__()
        self.name = name
        layer_sizes = [input_size] + hidden_sizes + [output_size]
        layers = []
        
        # Creazione dinamica dei layer
        for i in range(len(layer_sizes)-1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            if i < len(layer_sizes) - 2:
                layers.append(nn.ReLU()) # Funzione di attivazione ReLU per i layer nascosti
        self.layers = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.layers(x.flatten(1)) # Appiattisce l'input prima del passaggio nei layer

# Funzione per caricare e preparare i dati
def load_data(batch_size=128, val_size=5000, num_workers=0):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)) # Normalizzazione standard per MNIST
    ])
    
    ds_train = MNIST(root='./data', train=True, download=True, transform=transform)
    ds_test = MNIST(root='./data', train=False, download=True, transform=transform)
    
    # Divisione del dataset di training in train e validation
    indices = np.random.permutation(len(ds_train))
    ds_val, ds_train = Subset(ds_train, indices[:val_size]), Subset(ds_train, indices[val_size:])
    
    return (
        DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        DataLoader(ds_val, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        DataLoader(ds_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    )

# Classe per gestire il training e la valutazione del modello
class Trainer:
    def __init__(self, model, optimizer, device='cuda', use_tensorboard=True):
        self.model, self.optimizer, self.device = model, optimizer, device
        self.model.to(device)
        self.use_tensorboard = use_tensorboard
        if use_tensorboard:
            self.writer = SummaryWriter(f"logs/{model.name}")
            print(f"Tensorboard logs: logs/{model.name}")
        
    # Funzione per addestrare il modello per una singola epoca
    def train_epoch(self, dataloader, epoch):
        self.model.train()
        losses, correct, total = [], 0, 0
        
        for data, target in tqdm(dataloader, desc=f'Epoch {epoch}'):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = F.cross_entropy(output, target) # Loss Cross entropy
            loss.backward()
            self.optimizer.step()
            
            losses.append(loss.item())
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
        return np.mean(losses), 100. * correct / total
    
    # Funzione per valutare il modello sui dati di validazione o test
    def evaluate(self, dataloader, get_report=False):
        self.model.eval()
        losses, correct, total = [], 0, 0
        all_preds, all_targets = [], []
        
        with torch.no_grad():
            for data, target in tqdm(dataloader, desc='Evaluating'):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = F.cross_entropy(output, target)
                losses.append(loss.item())
                
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
                
                if get_report:
                    all_preds.extend(pred.cpu().numpy())
                    all_targets.extend(target.cpu().numpy())
        
        accuracy = 100. * correct / total
        if get_report:
            return np.mean(losses), accuracy, classification_report(all_targets, all_preds, digits=4)
        return np.mean(losses), accuracy
    
    # Funzione principale di training del modello
    def train(self, train_loader, val_loader, epochs=100):
        train_losses, train_accs, val_losses, val_accs = [], [], [], []
        
        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch(train_loader, epoch)
            val_loss, val_acc = self.evaluate(val_loader)
            
            train_losses.append(train_loss)
            train_accs.append(train_acc)
            val_losses.append(val_loss)
            val_accs.append(val_acc)
            
            print(f'Epoch {epoch}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
                  f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            
            if self.use_tensorboard:
                self.writer.add_scalar('Loss/train', train_loss, epoch)
                self.writer.add_scalar('Loss/val', val_loss, epoch)
                self.writer.add_scalar('Accuracy/train', train_acc, epoch)
                self.writer.add_scalar('Accuracy/val', val_acc, epoch)
        
        return train_losses, train_accs, val_losses, val_accs
    
    # Funzione per testare il modello sul set di test
    def test(self, test_loader):
        test_loss, test_acc, report = self.evaluate(test_loader, get_report=True)
        print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%')
        print("\nClassification Report:")
        print(report)
        return test_loss, test_acc, report


def main():
    # Variabili
    seed=1023
    batch_size, epochs, learning_rate = 256, 50, 0.0001
    
    # Imposto un seme per la riproducibilità
    torch.manual_seed(seed)
    np.random.seed(seed)

    #Verifico la presenza di gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Il training verrà eseguito su: {device}")
    
    # Caricamento dati
    train_loader, val_loader, test_loader = load_data(batch_size)
    
    # Modello e ottimizzatore
    model = MLP(input_size=28*28, hidden_sizes=[16, 16], output_size=10)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training
    trainer = Trainer(model, optimizer, device=device, use_tensorboard=True)
    train_losses, train_accs, val_losses, val_accs = trainer.train(
        train_loader, val_loader, epochs=epochs)
    
    # Test
    trainer.test(test_loader)
    
    # Visualizzazione
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(val_accs, label='Validation Accuracy')
    plt.title('Accuracy Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
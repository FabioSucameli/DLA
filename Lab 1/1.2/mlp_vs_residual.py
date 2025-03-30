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

# Blocco residuale per la ResidualMLP
class ResidualBlock(nn.Module):
    def __init__(self, size, layers=1):
        super().__init__()
        
        # Creo i layer interni al blocco
        self.layers = nn.ModuleList()
        for _ in range(layers):
            self.layers.append(nn.Linear(size, size))
            
    def forward(self, x):
        identity = x
        
        # Passa attraverso i layer interni con ReLU tra di essi
        out = x
        for i, layer in enumerate(self.layers):
            out = layer(out)
            if i < len(self.layers) - 1:  # Non applicare ReLU all'ultimo layer
                out = F.relu(out)
        
        # Somma l'output con l'input e applica ReLU
        return F.relu(out + identity)

    
# Modello MLP con connessioni residuali
class ResidualMLP(nn.Module):
    def __init__(self, input_size=784, hidden_size=16, depth=4, output_size=10, layers_per_block=1, name="ResidualMLP"):
        super().__init__()
        self.name = name
        self.input_layer = nn.Linear(input_size, hidden_size)
        
        # Creo i blocchi residuali
        self.residual_blocks = nn.ModuleList()
        for _ in range(depth-1):
            self.residual_blocks.append(ResidualBlock(hidden_size, layers_per_block))
            
        # Layer finale per la classificazione
        self.output_layer = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = x.flatten(1)    # appiattisco
        x = F.relu(self.input_layer(x)) # applico il layer di input
        
        # Passa attraverso i blocchi residuali
        for block in self.residual_blocks:
            x = block(x)
            
        return self.output_layer(x)

# Modello MLP (invariato)
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
                layers.append(nn.ReLU())
        self.layers = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.layers(x.flatten(1))


# Funzione per caricare e preparare i dati (invariata)
def load_data(batch_size=128, val_size=5000, num_workers=0):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
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

# Classe per gestire il training e la valutazione del modello (invariata)
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
    
    def test(self, test_loader):
        test_loss, test_acc, report = self.evaluate(test_loader, get_report=True)
        print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%')
        print("\nClassification Report:")
        print(report)
        return test_loss, test_acc, report


# Funzione per visualizzare il confronto tra MLP e ResidualMLP
def plot_comparison(train_losses_mlp, train_accs_mlp, val_losses_mlp, val_accs_mlp,
                   train_losses_res, train_accs_res, val_losses_res, val_accs_res, depth):
    plt.figure(figsize=(12, 10))
    
    # Loss curve
    plt.subplot(2, 2, 1)
    plt.plot(train_losses_mlp, label='MLP Train Loss')
    plt.plot(val_losses_mlp, label='MLP Validation Loss')
    plt.title(f'Loss Curves - Depth {depth}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(2, 2, 2)
    plt.plot(train_losses_res, label='ResidualMLP Train Loss')
    plt.plot(val_losses_res, label='ResidualMLP Validation Loss')
    plt.title(f'Loss Curves - Depth {depth}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Accuracy curve
    plt.subplot(2, 2, 3)
    plt.plot(train_accs_mlp, label='MLP Train Accuracy')
    plt.plot(val_accs_mlp, label='MLP Validation Accuracy')
    plt.title(f'Accuracy Curves - Depth {depth}')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.subplot(2, 2, 4)
    plt.plot(train_accs_res, label='ResidualMLP Train Accuracy')
    plt.plot(val_accs_res, label='ResidualMLP Validation Accuracy')
    plt.title(f'Accuracy Curves - Depth {depth}')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'results/comparison_depth_{depth}.png')
    plt.show()

# Funzione per visualizzare il confronto finale
def plot_final_comparison(results, depths):
    plt.figure(figsize=(15, 10))
    
    # Test accuracy
    plt.subplot(2, 2, 1)
    mlp_test_acc = [results['mlp'][d]['test_acc'] for d in depths]
    res_test_acc = [results['residual_mlp'][d]['test_acc'] for d in depths]
    plt.plot(depths, mlp_test_acc, 'o-', label='MLP')
    plt.plot(depths, res_test_acc, 'o-', label='ResidualMLP')
    plt.title('Test Accuracy vs Depth')
    plt.xlabel('Depth')
    plt.ylabel('Test Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    # Test loss
    plt.subplot(2, 2, 2)
    mlp_test_loss = [results['mlp'][d]['test_loss'] for d in depths]
    res_test_loss = [results['residual_mlp'][d]['test_loss'] for d in depths]
    plt.plot(depths, mlp_test_loss, 'o-', label='MLP')
    plt.plot(depths, res_test_loss, 'o-', label='ResidualMLP')
    plt.title('Test Loss vs Depth')
    plt.xlabel('Depth')
    plt.ylabel('Test Loss')
    plt.legend()
    plt.grid(True)
    
    # Final validation accuracy
    plt.subplot(2, 2, 3)
    mlp_val_acc = [results['mlp'][d]['val_accs'][-1] for d in depths]
    res_val_acc = [results['residual_mlp'][d]['val_accs'][-1] for d in depths]
    plt.plot(depths, mlp_val_acc, 'o-', label='MLP')
    plt.plot(depths, res_val_acc, 'o-', label='ResidualMLP')
    plt.title('Final Validation Accuracy vs Depth')
    plt.xlabel('Depth')
    plt.ylabel('Validation Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    # Final validation loss
    plt.subplot(2, 2, 4)
    mlp_val_loss = [results['mlp'][d]['val_losses'][-1] for d in depths]
    res_val_loss = [results['residual_mlp'][d]['val_losses'][-1] for d in depths]
    plt.plot(depths, mlp_val_loss, 'o-', label='MLP')
    plt.plot(depths, res_val_loss, 'o-', label='ResidualMLP')
    plt.title('Final Validation Loss vs Depth')
    plt.xlabel('Depth')
    plt.ylabel('Validation Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'results/depth_comparison.png')
    plt.show()

# Funzione principale
def main():
    # Variabili
    seed = 1023
    depths = [10,20,30]  # Profondità da testare
    batch_size, epochs, learning_rate = 256, 50, 0.0001
    hidden_size=32
    layers_per_block=1
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Verifico la presenza di GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Il training verrà eseguito su: {device}")
    
    # Caricamento dati
    train_loader, val_loader, test_loader = load_data(batch_size)
    
    # Risultati
    results = {'mlp': {}, 'residual_mlp': {}}
    
    for depth in depths:
        print(f"\nEsperimento con profondità: {depth}\n")
        
        # Configurazione MLP standard
        print("\nTraining MLP standard...")
        hidden_sizes = [hidden_size] * depth
        mlp = MLP(input_size=28*28, hidden_sizes=hidden_sizes, output_size=10, name=f"MLP_depth{depth}")
        optimizer_mlp = torch.optim.Adam(mlp.parameters(), learning_rate)
        trainer_mlp = Trainer(mlp, optimizer_mlp, device=device)
        
        # Training MLP standard
        train_losses_mlp, train_accs_mlp, val_losses_mlp, val_accs_mlp = trainer_mlp.train(
            train_loader, val_loader, epochs=epochs)
        
        # Test MLP standard
        test_loss_mlp, test_acc_mlp, _ = trainer_mlp.test(test_loader)
        
        # Configurazione ResidualMLP
        print("\nTraining ResidualMLP...")
        residual_mlp = ResidualMLP(input_size=28*28, hidden_size=hidden_size, depth=depth, 
                                  output_size=10, layers_per_block=layers_per_block, 
                                  name=f"ResidualMLP_depth{depth}")
        
        optimizer_res = torch.optim.Adam(residual_mlp.parameters(), learning_rate)
        trainer_res = Trainer(residual_mlp, optimizer_res, device=device)
        
        # Training ResidualMLP
        train_losses_res, train_accs_res, val_losses_res, val_accs_res = trainer_res.train(
            train_loader, val_loader, epochs=epochs)
        
        # Test ResidualMLP
        test_loss_res, test_acc_res, _ = trainer_res.test(test_loader)
        
        # Salvataggio risultati
        results['mlp'][depth] = {
            'train_losses': train_losses_mlp,
            'train_accs': train_accs_mlp,
            'val_losses': val_losses_mlp,
            'val_accs': val_accs_mlp,
            'test_loss': test_loss_mlp,
            'test_acc': test_acc_mlp
        }
        
        results['residual_mlp'][depth] = {
            'train_losses': train_losses_res,
            'train_accs': train_accs_res,
            'val_losses': val_losses_res,
            'val_accs': val_accs_res,
            'test_loss': test_loss_res,
            'test_acc': test_acc_res
        }
        
        # Visualizzazione confronto per questa profondità
        plot_comparison(train_losses_mlp, train_accs_mlp, val_losses_mlp, val_accs_mlp,
                        train_losses_res, train_accs_res, val_losses_res, val_accs_res, depth)
        
        if depth == 30:
            # Ottieni un batch di dati
            data, target = next(iter(train_loader))
            data, target = data.to(device), target.to(device)
                    
            # Calcola i gradienti per ResidualMLP
            output = residual_mlp(data)
            loss = F.cross_entropy(output, target)
            residual_mlp.zero_grad()
            loss.backward()
            weights_res = [p.grad.detach().cpu() for p in residual_mlp.parameters() if p.dim() == 2]

            # Calcola i gradienti per MLP standard
            output = mlp(data)
            loss = F.cross_entropy(output, target)
            mlp.zero_grad()
            loss.backward()
            weights_mlp = [p.grad.detach().cpu() for p in mlp.parameters() if p.dim() == 2]
                    
            # Solo il grafico di confronto diretto
            plt.figure(figsize=(10, 6))
            plt.plot([(p * p).sum().sqrt().item() for p in weights_mlp], 'o-', label='MLP standard')
            plt.plot([(p * p).sum().sqrt().item() for p in weights_res], 'o-', label='ResidualMLP')
            plt.xlabel('Layer')
            plt.ylabel('$||\\nabla_p \\mathcal{L}||$')
            plt.title(f'Confronto norma del gradiente dei pesi - Profondità {depth}')
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.savefig(f'results/gradient_comparison_depth_{depth}.png')
            plt.show()

    # Visualizzazione finale dei risultati per test accuracy
    plot_final_comparison(results, depths)
    

if __name__ == '__main__':
    main()
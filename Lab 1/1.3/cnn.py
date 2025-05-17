import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torchvision.models.resnet import BasicBlock
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
import os

# CNN standard (senza connessioni residuali)
class StandardCNN(nn.Module):
    def __init__(self, depth=10, num_classes=10):
        super().__init__()

        # Prima convoluzione mantiene la dimensione dell'immagine
        layers = []
        in_channels, out_channels = 3, 64

        # Prima convoluzione
        layers += [nn.Conv2d(in_channels, out_channels, 3, padding=1),
                   nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)]
        
        # Aggiunta layer convoluzionali in base alla profondità
        for i in range(1, depth):
            if i % 3 == 0 and i < depth - 1: # Ogni 3 layer, dimezza la dimensione
                layers += [nn.Conv2d(out_channels, out_channels*2, 3, stride=2, padding=1)]
                out_channels *= 2
            else:
                layers += [nn.Conv2d(out_channels, out_channels, 3, padding=1)]
            layers += [nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)]

        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.classifier = nn.Linear(out_channels, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

# CNN con connessioni residuali
class ResidualCNN(nn.Module):
    def __init__(self, depth=10, num_classes=10):
        super().__init__()
        self.in_planes = 64

        # Prima convoluzione
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        # Calcola il numero di blocchi per ogni layer
        num_blocks = (depth - 2) // 6  # -2 per conv1 e fc, //6 perché ogni BasicBlock ha 2 conv
        rest = (depth - 2) % 6
        
        # Crea i layer di blocchi residuali
        self.layer1 = self._make_layer(64, num_blocks + (1 if rest > 0 else 0), stride=1)
        self.layer2 = self._make_layer(128, num_blocks + (1 if rest > 2 else 0), stride=2)
        self.layer3 = self._make_layer(256, num_blocks + (1 if rest > 4 else 0), stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)

    def _make_layer(self, planes, num_blocks, stride):
        layers = []
        # Il primo blocco potrebbe avere stride=2 e cambiare il numero di canali
        downsample = None
        if stride != 1 or self.in_planes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )
        
        layers.append(BasicBlock(self.in_planes, planes, stride, downsample))
        self.in_planes = planes
        
        # I blocchi successivi hanno tutti stride=1 e stesso numero di canali
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(self.in_planes, planes, 1))
            
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        return self.fc(out)

# Funzione per caricare CIFAR-10
def load_cifar10_data(batch_size=128, num_workers=4):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    trainset = CIFAR10('./data', train=True, download=True, transform=transform_train)
    testset = CIFAR10('./data', train=False, download=True, transform=transform_test)

    # Split train set nel train e validation    
    train_dataset, val_dataset = torch.utils.data.random_split(trainset, [45000, 5000])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader, test_loader

# Trainer riutilizzata con integrazione W&B
class Trainer:
    def __init__(self, model, optimizer, device):
        self.model, self.optimizer = model.to(device), optimizer
        self.device = device
        wandb.watch(self.model, log='all', log_freq=100)

    def train_epoch(self, loader, epoch):
        self.model.train()
        total, correct, running_loss = 0, 0, 0.0

        for data, target in tqdm(loader, desc=f'Epoch {epoch}'):
            data, target = data.to(self.device), target.to(self.device)
        
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            self.optimizer.step()
        
            running_loss += loss.item() * data.size(0)
            preds = output.argmax(dim=1)
            correct += preds.eq(target).sum().item()
            total += data.size(0)
        
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        
        wandb.log({'train/loss': epoch_loss, 'train/accuracy': epoch_acc, 'epoch': epoch})
        return epoch_loss, epoch_acc

    def evaluate(self, loader, epoch):
        self.model.eval()
        total, correct, running_loss = 0, 0, 0.0
        with torch.no_grad():
            for data, target in loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = F.cross_entropy(output, target)
                running_loss += loss.item() * data.size(0)
                preds = output.argmax(dim=1)
                correct += preds.eq(target).sum().item()
                total += data.size(0)
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        wandb.log({'val/loss': epoch_loss, 'val/accuracy': epoch_acc, 'epoch': epoch})
        return epoch_loss, epoch_acc

    def test(self, loader):
        loss, acc = self.evaluate(loader, -1)
        wandb.log({'test/loss': loss, 'test/accuracy': acc})
        return loss, acc

# Funzione principale
def main():
    os.makedirs('models', exist_ok=True)
    # Parametri
    seed = 1023
    depths = [10, 20, 30] # Profondità da testare

    #In base alle profondità utilizzo una differente batch size per evitare OOM
    batch_map = { 
        10: 256,
        20: 128,
        30: 64
    }
    epochs, lr = 30, 1e-3

    # Verifica GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(seed)
    np.random.seed(seed)

    for depth in depths:
        batch_size = batch_map.get(depth, 128)  # valore di default se non specificato

        # Carica CIFAR-10
        train_loader, val_loader, test_loader = load_cifar10_data(batch_size)

        # Train & Validate Standard CNN
        run_name = f'standard_depth{depth}'
        wandb.init(project='CIFAR10-CNN', name=run_name, config={
            'model_type': 'standard',
            'depth': depth, 
            'epochs': epochs,
            'batch_size': batch_size, 
            'learning_rate': lr
        })

        std_model = StandardCNN(depth)
        optimizer = torch.optim.Adam(std_model.parameters(), lr=lr)
        trainer = Trainer(std_model, optimizer, device)

        for e in range(epochs): 
            trainer.train_epoch(train_loader, e)
            trainer.evaluate(val_loader, e)

        # Test Standard CNN
        trainer.test(test_loader)
        torch.save(std_model.state_dict(), f'models/standard_depth{depth}.pt')
        wandb.finish()

        # Train & Validate Residual CNN
        run_name = f'residual_depth{depth}'
        wandb.init(project='CIFAR10-CNN', name=run_name, config={
            'model_type': 'residual',
            'depth': depth, 
            'epochs': epochs,
            'batch_size': batch_size, 
            'learning_rate': lr
        })

        res_model = ResidualCNN(depth)
        optimizer = torch.optim.Adam(res_model.parameters(), lr=lr)
        trainer = Trainer(res_model, optimizer, device)

        for e in range(epochs): 
            trainer.train_epoch(train_loader, e)
            trainer.evaluate(val_loader, e)

        # Test Residual CNN
        trainer.test(test_loader)
        torch.save(res_model.state_dict(), f'models/residual_depth{depth}.pt')
        wandb.finish()



if __name__ == '__main__':
    main()
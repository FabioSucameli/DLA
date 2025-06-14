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

# Nuova classe: Knowledge Distillation Loss
class DistillationLoss(nn.Module):
    def __init__(self, alpha=0.3, temperature=4.0):
        super().__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.ce_loss = nn.CrossEntropyLoss()
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
    
    def forward(self, student_logits, teacher_logits, targets):
        # Hard targets loss (standard cross-entropy)
        hard_loss = self.ce_loss(student_logits, targets)
        
        # Soft targets loss (divergenza KL tra teacher e student)
        soft_student = F.log_softmax(student_logits / self.temperature, dim=1)
        soft_teacher = F.softmax(teacher_logits / self.temperature, dim=1)
        soft_loss = self.kl_loss(soft_student, soft_teacher) * (self.temperature ** 2)
        
        # Weighted combination
        total_loss = self.alpha * hard_loss + (1 - self.alpha) * soft_loss
        return total_loss, hard_loss, soft_loss

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

# TRAINER per Knowledge Distillation
class DistillationTrainer:
    def __init__(self, student_model, teacher_model, optimizer, device, alpha=0.3, temperature=4.0):
        self.student = student_model.to(device)
        self.teacher = teacher_model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.criterion = DistillationLoss(alpha, temperature)

        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False
            
        wandb.watch(self.student, log='all', log_freq=100)

    def train_epoch(self, loader, epoch):
        self.student.train()
        total, correct, running_loss = 0, 0, 0.0
        running_hard_loss, running_soft_loss = 0.0, 0.0

        for data, target in tqdm(loader, desc=f'Distillation Epoch {epoch}'):
            data, target = data.to(self.device), target.to(self.device)
        
            self.optimizer.zero_grad()
            
            # Ottengo le previsioni da entrambi i modelli
            student_logits = self.student(data)
            with torch.no_grad():
                teacher_logits = self.teacher(data)
            
            # Calcolo la distillation loss
            total_loss, hard_loss, soft_loss = self.criterion(student_logits, teacher_logits, target)
            total_loss.backward()
            self.optimizer.step()
        
            running_loss += total_loss.item() * data.size(0)
            running_hard_loss += hard_loss.item() * data.size(0)
            running_soft_loss += soft_loss.item() * data.size(0)
            
            preds = student_logits.argmax(dim=1)
            correct += preds.eq(target).sum().item()
            total += data.size(0)
        
        epoch_loss = running_loss / total
        epoch_hard_loss = running_hard_loss / total
        epoch_soft_loss = running_soft_loss / total
        epoch_acc = correct / total
        
        wandb.log({
            'train/total_loss': epoch_loss,
            'train/hard_loss': epoch_hard_loss,
            'train/soft_loss': epoch_soft_loss,
            'train/accuracy': epoch_acc,
            'epoch': epoch
        })
        return epoch_loss, epoch_acc

    def evaluate(self, loader, epoch, split='val'):
        self.student.eval()
        total, correct, running_loss = 0, 0, 0.0
        with torch.no_grad():
            for data, target in loader:
                data, target = data.to(self.device), target.to(self.device)
                student_logits = self.student(data)
                teacher_logits = self.teacher(data)
                
                total_loss, _, _ = self.criterion(student_logits, teacher_logits, target)
                running_loss += total_loss.item() * data.size(0)
                
                preds = student_logits.argmax(dim=1)
                correct += preds.eq(target).sum().item()
                total += data.size(0)

        epoch_loss = running_loss / total
        epoch_acc = correct / total

        log_dict = {
            f'{split}/loss': epoch_loss,
            f'{split}/accuracy': epoch_acc
        }
        if epoch >= 0:
            log_dict['epoch'] = epoch

        wandb.log(log_dict)
        return epoch_loss, epoch_acc

    def test(self, loader):
        return self.evaluate(loader, epoch=-1, split='test')

# Trainer originale per teacher e baseline
class Trainer:
    def __init__(self, model, optimizer, device, use_wandb=True):
        self.model, self.optimizer = model.to(device), optimizer
        self.device = device
        self.use_wandb = use_wandb

        if self.use_wandb:
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

        if self.use_wandb:
            wandb.log({'train/loss': epoch_loss, 'train/accuracy': epoch_acc, 'epoch': epoch})

        return epoch_loss, epoch_acc

    def evaluate(self, loader, epoch, split='val'):
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

        if self.use_wandb:
            log_dict = {
                f'{split}/loss': epoch_loss,
                f'{split}/accuracy': epoch_acc
            }
            if epoch >= 0:
                log_dict['epoch'] = epoch
            wandb.log(log_dict)

        return epoch_loss, epoch_acc

    def test(self, loader):
        return self.evaluate(loader, epoch=-1, split='test')


# main
# Si allena un modello teacher profondo. Se esiste già il modello salvato, viene caricato e testato direttamente per evitare il retrain
# Si allena uno student più piccolo con circa la metà dei parametri che viene prima allenato solo con etichette vere.
# Successivamente, viene riaddestrato usando sia le hard labels che le soft labels generate dal teacher.
# Si confrontano le performance per valutare l’efficacia della distillazione.
def main():
    os.makedirs('models', exist_ok=True)
    
    # Parametri
    seed = 1023
    teacher_depth = 30  # Teacher: modello profondo
    student_depth = 15  # Student: modello shallow
    batch_size = 128
    epochs, lr = 30, 1e-3
    
    # Parametri Knowledge Distillation
    alpha = 0.3  # Peso tra target “hard” e target “soft”
    temperature = 4.0  # Temperatura per softmax

    # Verifica GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Carica CIFAR-10
    train_loader, val_loader, test_loader = load_cifar10_data(batch_size)

    # Train Teacher Model
    teacher_path = f'models/teacher_residual_depth{teacher_depth}.pt'
    
    if not os.path.exists(teacher_path):
        print(f"Training Teacher Model (depth={teacher_depth})...")
        wandb.init(project='CIFAR10-KnowledgeDistillation', name=f'teacher_depth{teacher_depth}', config={
            'model_type': 'teacher_residual',
            'depth': teacher_depth,
            'epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': lr
        })

        teacher_model = ResidualCNN(teacher_depth)
        print(f"Teacher parameters: {sum(p.numel() for p in teacher_model.parameters())}")
        optimizer = torch.optim.Adam(teacher_model.parameters(), lr=lr)
        trainer = Trainer(teacher_model, optimizer, device)

        for e in range(epochs):
            trainer.train_epoch(train_loader, e)
            trainer.evaluate(val_loader, e)

        teacher_test_loss, teacher_test_acc = trainer.test(test_loader)
        print(f"Teacher Test Accuracy: {teacher_test_acc:.4f}")
        
        torch.save(teacher_model.state_dict(), teacher_path)
        wandb.finish()
    else:
        print(f"Loading existing teacher model from {teacher_path}")
        teacher_model = ResidualCNN(teacher_depth)
        teacher_model.load_state_dict(torch.load(teacher_path))

    # Train Student Model senza Knowledge Distillation (baseline)
    print(f"Training Student Baseline (depth={student_depth})...")
    wandb.init(project='CIFAR10-KnowledgeDistillation', name=f'student_baseline_depth{student_depth}', config={
        'model_type': 'student_baseline',
        'depth': student_depth,
        'epochs': epochs,
        'batch_size': batch_size,
        'learning_rate': lr
    })

    student_baseline = ResidualCNN(student_depth)
    print(f"Student parameters: {sum(p.numel() for p in student_baseline.parameters())}")
    optimizer = torch.optim.Adam(student_baseline.parameters(), lr=lr)
    trainer = Trainer(student_baseline, optimizer, device)

    for e in range(epochs):
        trainer.train_epoch(train_loader, e)
        trainer.evaluate(val_loader, e)

    baseline_test_loss, baseline_test_acc = trainer.test(test_loader)
    print(f"Student Baseline Test Accuracy: {baseline_test_acc:.4f}")
    
    torch.save(student_baseline.state_dict(), f'models/student_baseline_depth{student_depth}.pt')
    wandb.finish()

    # Train Student Model con Knowledge Distillation
    print(f"Training Student with Knowledge Distillation (depth={student_depth})...")
    wandb.init(project='CIFAR10-KnowledgeDistillation', name=f'student_distilled_depth{student_depth}', config={
        'model_type': 'student_distilled',
        'teacher_depth': teacher_depth,
        'student_depth': student_depth,
        'epochs': epochs,
        'batch_size': batch_size,
        'learning_rate': lr,
        'alpha': alpha,
        'temperature': temperature
    })

    student_distilled = ResidualCNN(student_depth)
    optimizer = torch.optim.Adam(student_distilled.parameters(), lr=lr)
    distill_trainer = DistillationTrainer(student_distilled, teacher_model, optimizer, device, alpha, temperature)

    for e in range(epochs):
        distill_trainer.train_epoch(train_loader, e)
        distill_trainer.evaluate(val_loader, e)

    distilled_test_loss, distilled_test_acc = distill_trainer.test(test_loader)
    print(f"Student Distilled Test Accuracy: {distilled_test_acc:.4f}")
    
    torch.save(student_distilled.state_dict(), f'models/student_distilled_depth{student_depth}.pt')
    wandb.finish()

    # Stampa i risultati di confronto
    print(f"Student Baseline (depth={student_depth}): {baseline_test_acc:.4f}")
    print(f"Student Distilled (depth={student_depth}): {distilled_test_acc:.4f}")
    print(f"Improvement from Distillation: {distilled_test_acc - baseline_test_acc:.4f}")

if __name__ == '__main__':
    main()

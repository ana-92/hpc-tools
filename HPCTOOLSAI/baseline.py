import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import pytorch_lightning as pl
import time

class LightningCIFAR10(pl.LightningModule):
    def __init__(self):
        super(LightningCIFAR10, self).__init__()

        # Definir el modelo
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

        # Loss function y optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        preds = torch.argmax(outputs, dim=1)
        acc = torch.sum(preds == labels).item() / (len(labels) * 1.0)
        self.log('train_loss', loss, on_step=True, on_epoch=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True)
        return loss
    def test_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        preds = torch.argmax(outputs, dim=1)
        acc = torch.sum(preds == labels).item() / (len(labels) * 1.0)
        self.log('test_loss', loss, on_step=True, on_epoch=True)
        self.log('test_acc', acc, on_step=True, on_epoch=True)
        return loss
    
    def configure_optimizers(self):
        return self.optimizer

# Transformaciones para normalizar las imágenes
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Descargar el conjunto de datos CIFAR-10
train_dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = CIFAR10(root='./data', train=False, download=True, transform=transform)

# Crear dataloaders
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

# Inicializar el módulo Lightning
lightning_model = LightningCIFAR10()

# Configuración de TensorBoard
writer = SummaryWriter("logs/CIFAR10")

# Configuración del entrenador de PyTorch Lightning
trainer = pl.Trainer(max_epochs=10, progress_bar_refresh_rate=50, gpus=1) 

# Entrenamiento con profiling
start_time = time.time()
trainer.fit(lightning_model, train_loader)
end_time = time.time()
total_time = end_time - start_time
minutes, seconds = divmod(total_time, 60)

# Guardar el tiempo total de entrenamiento en TensorBoard
writer.add_scalar("Total training time", total_time, 0)

#Imprimir el tiempo total del entrenamiento
print(f"Total Training Time: {int(minutes)} minutes {int(seconds)} seconds")

writer.flush()

# Evaluación en el conjunto de prueba
trainer.test(lightning_model, test_loader)

# Cierre del SummaryWriter
writer.close()
import os
import time
import copy
import torch
from tqdm import tqdm
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms

data_transforms = {
    "train": transforms.Compose([
        transforms.RandomRotation(5),
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(224, scale=(0.96, 1.0), ratio=(0.95, 1.05)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    "validation": transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

DATA_DIR = "dataset"
train_dir = os.path.join(DATA_DIR, "train")
val_dir = os.path.join(DATA_DIR, "validation")
image_datasets = {x: datasets.ImageFolder(os.path.join(DATA_DIR, x),
                                          data_transforms[x])
                  for x in ["train", "validation"]}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=32,
                                              shuffle=True, num_workers=4)
               for x in ["train", "validation"]}
dataset_sizes = {x: len(image_datasets[x]) for x in ["train", "validation"]}
class_names = image_datasets["train"].classes
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

for param in model.parameters():
    param.requires_grad = False

model.fc = nn.Linear(model.fc.in_features, 2)
model = model.to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

def train_model(_model, _criterion, _optimizer, _scheduler, _num_epochs):
    since = time.time()
    best_model_wts = copy.deepcopy(_model.state_dict())
    best_acc = 0.0

    for epoch in range(_num_epochs):
        print(f"Epoch {epoch + 1}/{_num_epochs}")
        print("-" * 10)

        for phase in ["train", "validation"]:
            if phase == "train":
                _model.train()
            else:
                _model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in tqdm(dataloaders[phase]):
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                _optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = _model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = _criterion(outputs, labels)

                    if phase == "train":
                        loss.backward()
                        _optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == "train":
                _scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            if phase == "validation" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(_model.state_dict())

        print()

    time_elapsed = time.time() - since

    print(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"Best val Acc: {best_acc:4f}")
    _model.load_state_dict(best_model_wts)

    return _model

model = train_model(model, criterion, optimizer, exp_lr_scheduler, _num_epochs=10)

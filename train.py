import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import os
import copy


def train_model():
    data_dir = "dataset"
    epochs = 5
    batch_size = 16

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{device.type.upper()}] Начинаем подготовку к обучению...")

    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    try:
        dataset = datasets.ImageFolder(data_dir, transform=data_transforms)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    except Exception as e:
        print(
            f"ОШИБКА: Не удалось прочитать папку 'dataset'. Убедитесь, что внутри есть папки 'real' и 'fake'.\nДетали: {e}")
        return

    print(f"Успех! Найдено {len(dataset)} изображений. Классы: {dataset.classes}")

    print("Загружаем базовую архитектуру ResNet18...")
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    print("\n--- СТАРТ ОБУЧЕНИЯ ---")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct_predictions = 0

        for step, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct_predictions += torch.sum(preds == labels.data)

            if step % 10 == 0 and step > 0:
                print(f"  Эпоха {epoch + 1} | Батч {step}/{len(dataloader)} обработан...")

        epoch_loss = running_loss / len(dataloader)
        epoch_acc = correct_predictions.double() / len(dataset)

        print(f"\n[ИТОГ ЭПОХИ {epoch + 1}/{epochs}] Ошибка: {epoch_loss:.4f} | Точность: {epoch_acc:.2%}")

        if epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())

    print(f"\n--- ОБУЧЕНИЕ ЗАВЕРШЕНО ---")
    print(f"Лучшая достигнутая точность: {best_acc:.2%}")

    model.load_state_dict(best_model_wts)

    save_path = "model_weights.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Лучшие веса успешно сохранены в файл: {save_path}")
    print("Теперь вы можете запустить main.py!")


if __name__ == "__main__":
    if not os.path.exists("dataset"):
        print("ОШИБКА: Папка 'dataset' не найдена! Создайте её в C:\\Work и положите туда папки 'real' и 'fake'.")
    else:
        train_model()
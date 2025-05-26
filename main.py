import os
import torch
import torchaudio
import torch.nn as nn
import torch.optim as optim
import wandb
import argparse
import torchinfo
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2
from torchvision.transforms.v2 import ToDtype
from torchvision import disable_beta_transforms_warning
disable_beta_transforms_warning()

# ─────── Constants ─────── #
SAMPLE_RATE = 22050
N_MFCC = 40
N_FFT = 1024
HOP_LENGTH = 512
MAX_TIME_STEPS = 400

# ─────── MixUp ─────── #
mixup = v2.MixUp(num_classes=10)

def mixup_collate_fn(batch):
    return mixup(*torch.utils.data.default_collate(batch))

def conf_collate_fn(mixup_enabled):
    return mixup_collate_fn if mixup_enabled else torch.utils.data.default_collate

# ─────── CNN Model ─────── #
class CustomCNN(nn.Module):
    def __init__(self):
        super(CustomCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 2)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

# ─────── Dataset ─────── #
class AudioDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.samples = []
        self.transform = transform
        self.label_map = {}

        for idx, class_name in enumerate(sorted(os.listdir(root_dir))):
            class_path = os.path.join(root_dir, class_name)
            if os.path.isdir(class_path):
                self.label_map[class_name] = idx
                for file_name in os.listdir(class_path):
                    if file_name.endswith('.wav'):
                        self.samples.append((os.path.join(class_path, file_name), idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path, label = self.samples[idx]
        waveform, _ = torchaudio.load(file_path)
        waveform = waveform.mean(dim=0, keepdim=True)

        mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            n_mels=64
        )(waveform)

        mel_spec = mel_spec[..., :MAX_TIME_STEPS]
        if mel_spec.shape[-1] < MAX_TIME_STEPS:
            pad_amt = MAX_TIME_STEPS - mel_spec.shape[-1]
            mel_spec = torch.nn.functional.pad(mel_spec, (0, pad_amt))

        mel_spec = mel_spec.unsqueeze(0)
        mel_spec = torch.nn.functional.interpolate(mel_spec, size=(128, 128), mode='bilinear', align_corners=False)
        mel_spec = mel_spec.squeeze(0).repeat(3, 1, 1)

        if self.transform:
            mel_spec = self.transform(mel_spec)

        return mel_spec, label

# ─────── Main Training Function ─────── #
def main(args):
    print("code started")
    run = wandb.init(project="deepfake-audio-spectrogram", config=vars(args))
    config = wandb.config

    collate_fn = conf_collate_fn(config.mixup)

    train_path = os.path.join(os.path.dirname(__file__), 'datasets/audio/train')
    test_path = os.path.join(os.path.dirname(__file__), 'datasets/audio/test')

    print("data loaded")

    train_data = AudioDataset(train_path)
    test_data = AudioDataset(test_path, transform=ToDtype(torch.float32, scale=True))

    print("data converted")

    train_loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_data, batch_size=config.batch_size, shuffle=False)

    model = CustomCNN()

    summary = torchinfo.summary(model, input_size=(1, 3, 128, 128))
    run.config['total_params'] = summary.total_params
    run.config['mult_adds'] = summary.total_mult_adds

    print("model begin training")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=config.learning_rate,
                          momentum=config.momentum, weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)

    max_accuracy = 0

    for epoch in range(config.epochs):
        model.train()
        total_train_loss = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        scheduler.step()
        avg_train_loss = total_train_loss / len(train_loader)

        model.eval()
        correct, total = 0, 0
        total_test_loss = 0.0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                total_test_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_test_loss = total_test_loss / len(test_loader)
        accuracy = correct / total
        max_accuracy = max(max_accuracy, accuracy)

        wandb.log({
            'epoch': epoch,
            'train_loss': avg_train_loss,
            'test_loss': avg_test_loss,
            'test_accuracy': accuracy,
            'max_accuracy': max_accuracy,
            'lr': optimizer.param_groups[0]['lr'],
        })

        print(f"[{epoch+1}/{config.epochs}] Accuracy: {accuracy:.4f}")

    torch.save(model.state_dict(), config.chkpt_path + run.id + '.pt')

# ─────── Entry Point ─────── #
if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-b", "--batch_size", type=int, default=16)
    parser.add_argument("-e", "--epochs", type=int, default=50)
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.01)
    parser.add_argument("-m", "--momentum", type=float, default=0.9)
    parser.add_argument("-wd", "--weight_decay", type=float, default=5e-4)
    parser.add_argument("--mixup", action="store_true")
    parser.add_argument("--depth", type=int, default=18)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--groups", type=int, default=1)
    parser.add_argument("--chkpt_path", type=str, default="./")
    args = parser.parse_args()
    main(args)

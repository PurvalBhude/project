import os
import torch
import torchaudio
import torch.nn as nn
import torch.optim as optim
import wandb
import argparse
import torchinfo
from torch.utils.data import Dataset, DataLoader

import numpy as np
import librosa
import cv2
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

# ─────── Constants ─────── #
SAMPLE_RATE = 22050
N_MFCC = 40
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
MAX_TIME_STEPS = 400

# ─────── CNN Model (Modified to remove one layer) ─────── #
class CustomCNN(nn.Module):
    def __init__(self):
        super(CustomCNN, self).__init__()
        self.features = nn.Sequential(
            # First Convolutional Block
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2), # Output: (32, 84, 200) from (1, 168, 400)

            # Second Convolutional Block (Now the last one before flattening)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2), # Output: (64, 42, 100)

            # The third convolutional block has been removed to make the model lighter.
            # nn.Conv2d(64, 128, kernel_size=3, padding=1),
            # nn.BatchNorm2d(128),
            # nn.ReLU(),
            # nn.MaxPool2d(2), # Original Output: (128, 21, 50)
        )
        self.classifier = nn.Sequential(
            # Adjust input features for Flatten based on the new output size from the last MaxPool2d
            # Now, the output from 'features' will be (64 channels * 42 height * 100 width)
            nn.Flatten(),
            nn.Linear(64 * 42 * 100, 256), # Updated linear layer input size
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 2)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

# ─────── Dataset ─────── #
class AudioDataset(Dataset):
    def __init__(self, root_dir):
        self.samples = []
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
        
        try:
            audio, sr = librosa.load(file_path, sr=SAMPLE_RATE)

            mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH)
            mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

            mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH)

            mel_spec = cv2.resize(mel_spec, (MAX_TIME_STEPS, N_MELS))
            mfcc = cv2.resize(mfcc, (MAX_TIME_STEPS, N_MFCC))

            mel_spec = (mel_spec - np.mean(mel_spec)) / np.std(mel_spec)
            mfcc = (mfcc - np.mean(mfcc)) / np.std(mfcc)

            combined_features = np.vstack((mel_spec, mfcc))
            combined_features = torch.from_numpy(combined_features).float().unsqueeze(0)

        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            combined_features = torch.zeros((1, N_MELS + N_MFCC, MAX_TIME_STEPS), dtype=torch.float32)
            label = -1 

        return combined_features, label

# ─────── Main Training Function ─────── #
def main(args):
    print("code started")
    run = wandb.init(project="deepfake-audio-spectrogram-mfcc-less-heavy", config=vars(args)) # Updated project name
    config = wandb.config
    
    script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
    train_path = os.path.join(script_dir, 'datasets/audio/train')
    test_path = os.path.join(script_dir, 'datasets/audio/test')

    print("data loading initiated")

    train_data = AudioDataset(train_path)
    test_data = AudioDataset(test_path)

    print("data loaded")
    
    train_loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=config.batch_size, shuffle=False)

    model = CustomCNN()

    # Input size for torchinfo summary: (batch_size, channels, height, width)
    # Height is N_MELS + N_MFCC
    summary = torchinfo.summary(model, input_size=(1, 1, N_MELS + N_MFCC, MAX_TIME_STEPS))
    run.config['total_params'] = summary.total_params
    run.config['mult_adds'] = summary.total_mult_adds
    print(f"Model Total Parameters: {summary.total_params}")
    print(f"Model Multiply-Adds: {summary.total_mult_adds}")


    print("model begin training")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-6) 
    
    best_val_loss = float('inf')
    patience_counter = 0
    early_stopping_patience = 5

    for epoch in range(config.epochs):
        model.train()
        total_train_loss = 0.0

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            if (labels == -1).any():
                print(f"Skipping batch {batch_idx} due to error in data loading.")
                continue

            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        model.eval()
        correct, total = 0, 0
        total_test_loss = 0.0
        all_labels = []
        all_predictions = []
        all_probabilities = []

        with torch.no_grad():
            for inputs, labels in test_loader:
                if (labels == -1).any():
                    print(f"Skipping batch {batch_idx} in evaluation due to error in data loading.")
                    continue

                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                total_test_loss += loss.item()
                
                probabilities = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)

                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())
                all_probabilities.extend(probabilities[:, 1].cpu().numpy())

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_test_loss = total_test_loss / len(test_loader)
        accuracy = correct / total

        scheduler.step(avg_test_loss)

        if avg_test_loss < best_val_loss:
            best_val_loss = avg_test_loss
            patience_counter = 0
            torch.save(model.state_dict(), config.chkpt_path + run.id + '_best_model.pt')
            print(f"[{epoch+1}/{config.epochs}] Validation loss improved. Saving model.")
        else:
            patience_counter += 1
            print(f"[{epoch+1}/{config.epochs}] Validation loss did not improve. Patience: {patience_counter}/{early_stopping_patience}")
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

        wandb.log({
            'epoch': epoch,
            'train_loss': avg_train_loss,
            'test_loss': avg_test_loss,
            'test_accuracy': accuracy,
            'lr': optimizer.param_groups[0]['lr'],
        })

        print(f"[{epoch+1}/{config.epochs}] Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}, Test Accuracy: {accuracy:.4f}")

    if os.path.exists(config.chkpt_path + run.id + '_best_model.pt'):
        model.load_state_dict(torch.load(config.chkpt_path + run.id + '_best_model.pt'))
        model.eval()
        print("Loaded best model for final evaluation.")
    
    y_true = np.array(all_labels)
    y_pred_classes = np.array(all_predictions)
    y_scores = np.array(all_probabilities)

    print("\n─────── Classification Report ───────")
    print(classification_report(y_true, y_pred_classes))

    print("\n─────── Confusion Matrix ───────")
    conf_matrix = confusion_matrix(y_true, y_pred_classes)
    plt.figure(figsize=(6,5))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    wandb.log({"confusion_matrix": wandb.Image(plt)})
    plt.show()
    plt.close()

    print("\n─────── ROC Curve ───────")
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    wandb.log({"roc_curve": wandb.Image(plt)})
    plt.show()
    plt.close()
    
    print(f"✅ Model AUC Score: {roc_auc:.4f}")
    wandb.log({'final_auc_score': roc_auc})
    
    wandb.finish()


# ─────── Entry Point ─────── #
if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-b", "--batch_size", type=int, default=16)
    parser.add_argument("-e", "--epochs", type=int, default=50)
    parser.add_argument("-lr", "--learning_rate", type=float, default=3e-4)
    parser.add_argument("-m", "--momentum", type=float, default=0.0)
    parser.add_argument("-wd", "--weight_decay", type=float, default=0.0)
    parser.add_argument("--depth", type=int, default=18)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--groups", type=int, default=1)
    parser.add_argument("--chkpt_path", type=str, default="./")
    args = parser.parse_args()
    main(args)
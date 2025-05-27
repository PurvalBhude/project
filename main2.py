import os
import torch
import torchaudio
import torch.nn as nn
import torch.optim as optim
import wandb
import argparse
import torchinfo
from torch.utils.data import Dataset, DataLoader
# from torchvision.transforms import v2 # Not needed for notebook conversion
# from torchvision.transforms.v2 import ToDtype # Not needed
# from torchvision import disable_beta_transforms_warning # Not needed
# disable_beta_transforms_warning() # Not needed

import numpy as np
import librosa
import cv2
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import gc # For garbage collection, if needed, though PyTorch memory management is different

# ─────── Constants ─────── #
SAMPLE_RATE = 22050
# N_MFCC = 40 # Not used in notebook
N_MELS = 128 # From notebook
N_FFT = 2048 # From notebook
HOP_LENGTH = 512
MAX_TIME_STEPS = 400 # From notebook (for resizing spectrogram width)

# ─────── MixUp (Removed as per notebook) ─────── #
# mixup = v2.MixUp(num_classes=10) # Removed

# def mixup_collate_fn(batch): # Removed
#     return mixup(*torch.utils.data.default_collate(batch)) # Removed

# def conf_collate_fn(mixup_enabled): # Removed
#     return mixup_collate_fn if mixup_enabled else torch.utils.data.default_collate # Removed

# ─────── CNN Model ─────── #
class CustomCNN(nn.Module):
    def __init__(self):
        super(CustomCNN, self).__init__()
        self.features = nn.Sequential(
            # Input channel changed from 3 to 1 to match notebook's single-channel spectrogram
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), # Batch norm after Conv, before ReLU (notebook style)
            nn.ReLU(),
            nn.MaxPool2d(2), # Output: (32, 64, 200)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2), # Output: (64, 32, 100)
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2), # Output: (128, 16, 50)
        )
        self.classifier = nn.Sequential(
            # Adjust input features for Flatten based on new (128, 400) input size
            # (128 channels * 16 height * 50 width)
            nn.Flatten(),
            nn.Linear(128 * 16 * 50, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 2) # Output for 2 classes
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

# ─────── Dataset ─────── #
class AudioDataset(Dataset):
    def __init__(self, root_dir): # Removed transform parameter
        self.samples = []
        # self.transform = transform # Removed
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
            # Load audio using librosa as in the notebook
            audio, sr = librosa.load(file_path, sr=SAMPLE_RATE)

            # Compute Mel Spectrogram as in the notebook
            mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH)
            mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

            # Resize to Fixed Time Dimension (MAX_TIME_STEPS) and N_MELS as in the notebook
            # cv2.resize expects (width, height)
            mel_spec = cv2.resize(mel_spec, (MAX_TIME_STEPS, N_MELS)) # (400, 128)

            # Normalize Features as in the notebook
            mel_spec = (mel_spec - np.mean(mel_spec)) / np.std(mel_spec)

            # Convert to PyTorch tensor and add channel dimension (1 channel)
            mel_spec = torch.from_numpy(mel_spec).float().unsqueeze(0) # Shape: (1, 128, 400)

        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            # Return a dummy tensor and label if loading fails to prevent DataLoader errors
            # In a real scenario, you might skip or handle this more robustly
            mel_spec = torch.zeros((1, N_MELS, MAX_TIME_STEPS), dtype=torch.float32)
            label = -1 # Indicate an error or skip in collate_fn if desired

        # No transformation needed as per notebook conversion
        # if self.transform:
        #     mel_spec = self.transform(mel_spec)

        return mel_spec, label

# ─────── Main Training Function ─────── #
def main(args):
    print("code started")
    run = wandb.init(project="deepfake-audio-spectrogram-notebook-conversion", config=vars(args))
    config = wandb.config

    # collate_fn = conf_collate_fn(config.mixup) # Removed MixUp
    
    # Define paths
    # Assuming 'datasets/audio/train' and 'datasets/audio/test' are relative to the script
    # Or, you might need to adjust these paths if your data is elsewhere
    script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
    train_path = os.path.join(script_dir, 'datasets/audio/train')
    test_path = os.path.join(script_dir, 'datasets/audio/test')

    print("data loading initiated")

    train_data = AudioDataset(train_path) # Removed transform
    test_data = AudioDataset(test_path) # Removed transform

    print("data loaded")

    # The notebook's stratified split would typically happen here on file paths
    # For simplicity, we'll keep PyTorch's default random split if needed for val set,
    # but the current structure uses separate train/test directories.
    # If a validation set is needed from the training data:
    # train_size = int(0.7 * len(train_data))
    # val_size = len(train_data) - train_size
    # train_subset, val_subset = torch.utils.data.random_split(train_data, [train_size, val_size])
    # train_loader = DataLoader(train_subset, batch_size=config.batch_size, shuffle=True)
    # val_loader = DataLoader(val_subset, batch_size=config.batch_size, shuffle=False)
    # For now, following the spirit of using a distinct 'test' folder as validation
    
    train_loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True) # collate_fn removed
    test_loader = DataLoader(test_data, batch_size=config.batch_size, shuffle=False)

    model = CustomCNN()

    # Input size for torchinfo summary: (batch_size, channels, height, width)
    summary = torchinfo.summary(model, input_size=(1, 1, N_MELS, MAX_TIME_STEPS))
    run.config['total_params'] = summary.total_params
    run.config['mult_adds'] = summary.total_mult_adds

    print("model begin training")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model = model.to(device)
    criterion = nn.CrossEntropyLoss() # Keras's categorical_crossentropy is equivalent for one-hot labels, PyTorch CrossEntropyLoss expects class indices
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate) # Changed to Adam
    
    # Learning Rate Scheduler and Early Stopping as per notebook
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-6, verbose=True)
    
    best_val_loss = float('inf')
    patience_counter = 0
    early_stopping_patience = 5 # From notebook's EarlyStopping

    for epoch in range(config.epochs):
        model.train()
        total_train_loss = 0.0

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels) # labels are class indices
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
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                total_test_loss += loss.item()
                
                probabilities = torch.softmax(outputs, dim=1) # Get probabilities for ROC AUC
                _, predicted = torch.max(outputs, 1) # Get predicted class for accuracy

                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())
                all_probabilities.extend(probabilities[:, 1].cpu().numpy()) # Probabilities for the positive class

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_test_loss = total_test_loss / len(test_loader)
        accuracy = correct / total

        # Learning Rate Scheduler step based on validation loss
        scheduler.step(avg_test_loss)

        # Early Stopping check
        if avg_test_loss < best_val_loss:
            best_val_loss = avg_test_loss
            patience_counter = 0
            # Save the best model
            torch.save(model.state_dict(), config.chkpt_path + run.id + '_best_model.pt')
            print(f"[{epoch+1}/{config.epochs}] Validation loss improved. Saving model.")
        else:
            patience_counter += 1
            print(f"[{epoch+1}/{config.epochs}] Validation loss did not improve. Patience: {patience_counter}/{early_stopping_patience}")
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break # Exit training loop

        wandb.log({
            'epoch': epoch,
            'train_loss': avg_train_loss,
            'test_loss': avg_test_loss,
            'test_accuracy': accuracy,
            'lr': optimizer.param_groups[0]['lr'],
        })

        print(f"[{epoch+1}/{config.epochs}] Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}, Test Accuracy: {accuracy:.4f}")

    # Load best model for final evaluation if early stopping occurred
    if os.path.exists(config.chkpt_path + run.id + '_best_model.pt'):
        model.load_state_dict(torch.load(config.chkpt_path + run.id + '_best_model.pt'))
        model.eval()
        print("Loaded best model for final evaluation.")
    
    # Final Evaluation Metrics (as in notebook)
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
    plt.close() # Close plot to prevent display issues in some environments

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
    plt.close() # Close plot
    
    print(f"✅ Model AUC Score: {roc_auc:.4f}")
    wandb.log({'final_auc_score': roc_auc})

    # The original code saved the last state, this saves the best if early stopping was used
    # torch.save(model.state_dict(), config.chkpt_path + run.id + '.pt') 
    
    wandb.finish()


# ─────── Entry Point ─────── #
if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-b", "--batch_size", type=int, default=16)
    parser.add_argument("-e", "--epochs", type=int, default=50)
    # Changed default LR to match notebook's Adam LR
    parser.add_argument("-lr", "--learning_rate", type=float, default=3e-4) 
    # Momentum and weight_decay are for SGD, removed for Adam or set to 0.
    # Adam has its own regularization mechanisms. Keeping them but setting to 0 for Adam defaults.
    parser.add_argument("-m", "--momentum", type=float, default=0.0) 
    parser.add_argument("-wd", "--weight_decay", type=float, default=0.0)
    # Mixup removed, so this argument is no longer needed
    # parser.add_argument("--mixup", action="store_true") 
    
    # Model depth/width arguments are not used in CustomCNN, but were in original code's argparse
    # Keeping them for compatibility if other models were intended to be used, but they are inert here.
    parser.add_argument("--depth", type=int, default=18) 
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--groups", type=int, default=1)
    
    parser.add_argument("--chkpt_path", type=str, default="./")
    args = parser.parse_args()
    main(args)
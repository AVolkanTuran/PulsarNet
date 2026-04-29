#Configure the file directories and hyperparameters at the top for ease of use
DATA_DIR         = "./S0_json" 
PULSAR_LIST      = "./known_pulsars.txt"
CACHE_FILE       = "./pulsar_dataset.npz"
CHECKPOINT_DIR   = "./checkpoints" 

BATCH_SIZE       = 128
NUM_EPOCHS       = 30
LEARNING_RATE    = 1e-3
WEIGHT_DECAY     = 1e-4
RANDOM_SEED      = 42

#Focal loss hyperparameters
FOCAL_ALPHA      = 0.25
FOCAL_GAMMA      = 2.0

#Classification threshold at the beginning, which gets tuned later on for better recall
DEFAULT_THRESHOLD = 0.5

import os
import json
import base64
import random
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.amp import autocast, GradScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    classification_report,
    precision_recall_curve,
)
from tqdm import tqdm


#We picked a random seed to be able to reproduce our results
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)


#Since we had a powerful GPU, Nvidia RTX 4080, we moved everything to the GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == "cuda":
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


#Our subints in json files had different sizes, so we had to specify a standard size for them in order to save them in the cache
SUBINTS_H = 32
SUBINTS_W = 512
DM_LEN    = 3084


#This is a function to resize our subints to the standard size by interpolating
def _resize_subints(subints_np):
    src = np.ascontiguousarray(subints_np, dtype=np.float32)
    return cv2.resize(src, (SUBINTS_W, SUBINTS_H), interpolation=cv2.INTER_AREA)

#The function to preprocess json files
def preprocess_json_files(data_dir, pulsar_list_path, cache_path):

    print("\nPreprocessing JSON files")

    #Load known pulsars
    with open(pulsar_list_path) as f:
        pulsar_set = set(line.strip() for line in f if line.strip())
    print(f"Loaded {len(pulsar_set)} known pulsars from {pulsar_list_path}")

    #Get a list of json files
    json_files = [f for f in os.listdir(data_dir) if f.endswith(".json")]
    n_files   = len(json_files)
    print(f"Found {n_files} JSON files in {data_dir}")

    #Each json file has three different types of information. a list of scalar values, a plot of subints, and a DM curve
    all_scalars  = np.zeros((n_files, 6), dtype=np.float32)
    all_subints  = np.zeros((n_files, SUBINTS_H, SUBINTS_W), dtype=np.float32)
    all_dmcurves = np.zeros((n_files, DM_LEN), dtype=np.float32)
    all_labels   = np.zeros(n_files, dtype=np.float32)

    #Keep track of the index we are writing to, and the number of skipped files
    write_idx = 0
    skipped   = 0

    #Add a progress bar to visualize the preprocessing step while looping through each file
    for fname in tqdm(json_files, desc="Reading JSONs"):
        fpath = os.path.join(data_dir, fname)
        stem  = fname.replace(".json", "")

        try:
            with open(fpath) as f:
                d = json.load(f)

            #Extract the scalar parameters
            p = d["params"]
            scalars = np.array([
                p["period"], p["freq"], p["dm"],
                p["width"], p["ducy"],  p["snr"],
            ], dtype=np.float32)

            #Extract the subints information and resize to standard (32,512)
            raw     = base64.b64decode(d["subints"]["data"])
            subints = np.frombuffer(raw, dtype=np.float32).reshape(d["subints"]["shape"])
            subints = _resize_subints(subints)

            #Extract the DM curve information and pad/truncate to 3084
            raw2     = base64.b64decode(d["peaks"]["values"]["data"])
            peaks    = np.frombuffer(raw2, dtype=np.float64).reshape(d["peaks"]["values"]["shape"])
            dm_curve = peaks[:, 5].astype(np.float32)
            if len(dm_curve) < DM_LEN:
                dm_curve = np.pad(dm_curve, (0, DM_LEN - len(dm_curve)))
            else:
                dm_curve = dm_curve[:DM_LEN]

            #If the pulsar is in the known_pulsars.txt, then label it as 1. If it is RFI, label it as 0.
            label = 1 if stem in pulsar_set else 0


            all_scalars[write_idx]  = scalars
            all_subints[write_idx]  = subints
            all_dmcurves[write_idx] = dm_curve
            all_labels[write_idx]   = label
            write_idx += 1

        #If there is an error, just skip the file
        except Exception:
            skipped += 1
            continue

    #Trim just in case any files were skipped
    all_scalars  = all_scalars[:write_idx]
    all_subints  = all_subints[:write_idx]
    all_dmcurves = all_dmcurves[:write_idx]
    all_labels   = all_labels[:write_idx]

    print(f"\nProcessed {write_idx} files, skipped {skipped}")
    print(f"  Pulsars : {int(all_labels.sum())}")
    print(f"  RFI     : {int((all_labels == 0).sum())}")

    #Save the preprocessed files to cache as this step takes a really long time (Took around 30 minutes for us).
    print(f"\nSaving cache to {cache_path} ...")
    np.savez_compressed(
        cache_path,
        scalars  = all_scalars,
        subints  = all_subints,
        dmcurves = all_dmcurves,
        labels   = all_labels,
    )
    print("Cache saved.")

#Function to load the data
def load_data(data_dir, pulsar_list_path, cache_path):
    #If the data is not preprocessed (i.e. not in the cache), preprocess the data
    if not os.path.exists(cache_path):
        preprocess_json_files(data_dir, pulsar_list_path, cache_path)

    #Load the data from cache
    print(f"\nLoading dataset from cache: {cache_path}")
    data = np.load(cache_path)
    scalars  = data["scalars"]
    subints  = data["subints"]
    dmcurves = data["dmcurves"]
    labels   = data["labels"]

    print(f"  Total samples : {len(labels)}")
    print(f"  Pulsars       : {int(labels.sum())}")
    print(f"  RFI           : {int((labels == 0).sum())}")
    print(f"  Scalars shape : {scalars.shape}")
    print(f"  Subints shape : {subints.shape}")
    print(f"  DM curve shape: {dmcurves.shape}")

    return scalars, subints, dmcurves, labels

#Compute the values requires for the normalization of the data before we train the model
def compute_normalization(scalars, subints, dmcurves):
    scalar_mean = scalars.mean(axis=0)
    scalar_std  = scalars.std(axis=0) + 1e-8  #avoid dividing by zero

    #Normalize subint row row across all phase bins
    subints_mean = subints.mean(axis=(0, 2), keepdims=True) 
    subints_std  = subints.std(axis=(0, 2),  keepdims=True) + 1e-8

    dm_mean = dmcurves.mean()
    dm_std  = dmcurves.std() + 1e-8

    return {
        "scalar_mean"  : scalar_mean,
        "scalar_std"   : scalar_std,
        "subints_mean" : subints_mean,
        "subints_std"  : subints_std,
        "dm_mean"      : dm_mean,
        "dm_std"       : dm_std,
    }


def apply_normalization(scalars, subints, dmcurves, norm):
    scalars  = (scalars  - norm["scalar_mean"])/norm["scalar_std"]
    subints  = (subints  - norm["subints_mean"])/norm["subints_std"]
    dmcurves = (dmcurves - norm["dm_mean"])/norm["dm_std"]
    return scalars, subints, dmcurves


#For training data, we want to augment subints data. We randomly phase-roll the data
#Shifted pulsars should still be classified as pulsars, so this is great for our training.
def augment_subints(subints):
    shift = np.random.randint(0, subints.shape[1])
    return np.roll(subints, shift, axis=1)

#Our custom pulsar pytorch dataset
class PulsarDataset(Dataset):
    def __init__(self, scalars, subints, dmcurves, labels, augment=False):
        self.scalars  = scalars
        self.subints  = subints
        self.dmcurves = dmcurves
        self.labels   = labels
        self.augment  = augment

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        scalars  = self.scalars[idx].copy()
        subints  = self.subints[idx].copy()
        dmcurve  = self.dmcurves[idx].copy()
        label    = self.labels[idx]

        if self.augment and label == 1:
            subints = augment_subints(subints)

        return (
            torch.tensor(scalars, dtype=torch.float32),
            torch.tensor(subints, dtype=torch.float32),
            torch.tensor(dmcurve, dtype=torch.float32),
            torch.tensor(label, dtype=torch.float32),
        )

#Our custom neural network (PulsarNet)
#It consists of three branches for each type of input in each json file.
#The scalar values go through an MLP branch. The subints go through a 2D convolutional branch.
#The DM curve values go through a 1D convolutional branch.
#All the branches are then combined and pushed through a few more layers.
class PulsarNet(nn.Module):
    
    def __init__(self):
        super().__init__()

        #The MLP branch
        self.mlp = nn.Sequential(
            nn.Linear(6, 64),
            nn.BatchNorm1d(64),
            #We used ReLU activation function because it is the standard right now
            #Prevents vanishing gradients and 
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 64),
            nn.ReLU(),
        )  #(64,)

        #The 2D CNN branch
        self.cnn2d = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(3, 7), padding=(1, 3)),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d((2, 4)),
            nn.Conv2d(16, 32, kernel_size=(3, 7), padding=(1, 3)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((2, 4)),
            nn.Conv2d(32, 64, kernel_size=(3, 5), padding=(1, 2)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 16)),
            nn.Flatten(), #(1024,)
        )

        #The 1D CNN branch
        self.cnn1d = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, padding=3),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(4),

            nn.Conv1d(16, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(4),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(16),
            nn.Flatten(), #(1024,)
        )

        #The final layers for the combined branches
        self.head = nn.Sequential(
            nn.Linear(64 + 1024 + 1024, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    #Forward-pass function that defines how the data is pushed through the network.
    def forward(self, scalars, subints, dmcurve):
        m  = self.mlp(scalars)
        c2 = self.cnn2d(subints.unsqueeze(1))
        c1 = self.cnn1d(dmcurve.unsqueeze(1))
        #Combine together the three branches
        fused = torch.cat([m, c2, c1], dim=1)
        return self.head(fused).squeeze(1)

#We decided to use a Focal Loss function. Although most neural networks we encountered used Cross-Entropy,
#we learned that at extreme class imbalances, Focal Loss is a better option since it account for the imbalance with an extra term.
#Our data has an extreme class imbalance, so we went with this function.
class FocalLoss(nn.Module):

    def __init__(self, alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        bce    = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        pt     = torch.exp(-bce)
        focal  = self.alpha * (1 - pt) ** self.gamma * bce
        return focal.mean()

#This function trains the model for one epoch.
def train_one_epoch(model, loader, optimizer, criterion, scaler):
    model.train()
    total_loss = 0.0

    for scalars, subints, dmcurve, labels in loader:
        #All the data is sent to the GPU for training
        scalars  = scalars.to(device)
        subints  = subints.to(device)
        dmcurve  = dmcurve.to(device)
        labels   = labels.to(device)

        optimizer.zero_grad()

        if device.type == 'cuda':
            with autocast('cuda'):
                logits = model(scalars, subints, dmcurve)
                loss   = criterion(logits, labels)
        else:
            with autocast('cpu'):
                logits = model(scalars, subints, dmcurve)
                loss   = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    return total_loss / len(loader)

#This function evaluates how well our model is doing
@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    all_logits = []
    all_labels = []

    for scalars, subints, dmcurve, labels in loader:
        scalars  = scalars.to(device)
        subints  = subints.to(device)
        dmcurve  = dmcurve.to(device)

        if device.type == 'cuda':
            with autocast('cuda'):
                logits = model(scalars, subints, dmcurve)
        else:
            with autocast('cpu'):
                logits = model(scalars, subints, dmcurve)

        all_logits.append(logits.cpu())
        all_labels.append(labels)

    logits = torch.cat(all_logits).float().numpy()
    labels = torch.cat(all_labels).numpy()
    #Sigmoid function puts the output logits between 0 and 1, which we can use to classify based on our threshold.
    scores = torch.sigmoid(torch.tensor(logits)).numpy()

    #Although auroc is not a good indicator of how well our model does because of the imbalance, we still kept it as it doesn't take much to computer
    #and it is kind of standard to calculate from what we have seen.
    auroc = roc_auc_score(labels, scores)
    #AP is much more important for our model. We want extremely high recall for our model, since it is fine if we have false positives.
    #We can just have humans go through them once again. Within that high recall requirement, we want to maximize our precision, so maximizing
    #AP helps us optimize a good balance between recall and precision, and find the maximum precision for our required recall.
    #AP is the area under the curve of a precision vs. recall curve
    ap    = average_precision_score(labels, scores)
    #If the sigmoid output is higher than the threshold, it is classified as a pulsar
    preds = (scores >= DEFAULT_THRESHOLD).astype(int)

    return auroc, ap, scores, labels, preds

#This function helps us find the optimal threshold that maximizes preceision 
#while keeping our recall above a min_recall (which is more important for us)
def find_optimal_threshold(labels, scores, min_recall=0.95):
    precisions, recalls, thresholds = precision_recall_curve(labels, scores)
    # Find indices where recall >= min_recall
    valid = np.where(recalls[:-1] >= min_recall)[0]
    if len(valid) == 0:
        print(f"Warning: could not achieve recall >= {min_recall}")
        return DEFAULT_THRESHOLD
    
    #Among valid thresholds, pick the one with highest precision
    best_idx       = valid[np.argmax(precisions[valid])]
    best_threshold = thresholds[best_idx]
    best_precision = precisions[best_idx]
    best_recall    = recalls[best_idx]
    print(f"\nOptimal threshold (recall >= {min_recall}): {best_threshold:.4f}")
    print(f"  Precision: {best_precision:.4f}  Recall: {best_recall:.4f}")
    return best_threshold

#This is the main function that puts everything above together
def main():
    #Create a directory for our best model
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    best_path = os.path.join(CHECKPOINT_DIR, "best_model.pt")

    #Load the data
    scalars, subints, dmcurves, labels = load_data(
        DATA_DIR, PULSAR_LIST, CACHE_FILE
    )

    #We used a stratified train/validation/test split
    #Stratification is necessary since our data is extremely imbalanced.
    #Validation split is used for each epoch and for threshold tuning.
    #80% Training, 10% Validation, 10% Test
    idx = np.arange(len(labels))

    idx_train, idx_temp, y_train, y_temp = train_test_split(
        idx, labels, test_size=0.2, stratify=labels, random_state=RANDOM_SEED
    )
    idx_val, idx_test, y_val, y_test = train_test_split(
        idx_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=RANDOM_SEED
    )

    print(f"\nSplit sizes:")
    print(f"  Train : {len(idx_train)} ({int(y_train.sum())} pulsars)")
    print(f"  Val   : {len(idx_val)}   ({int(y_val.sum())} pulsars)")
    print(f"  Test  : {len(idx_test)}  ({int(y_test.sum())} pulsars)")
    
    #If there is already a best model, ask the user to train or use the model.
    if os.path.exists(best_path):
        print(f"\nFound existing checkpoint at {best_path}")
        while True:
            answer = input("Use saved model (s) or retrain from scratch (r)? [s/r]: ").strip().lower()
            if answer in ("s", "r"):
                break
            print("  Please enter 's' or 'r'.")
        skip_training = (answer == "s")
    else:
        print(f"\nNo checkpoint found at {best_path} — will train from scratch.")
        skip_training = False

    if skip_training:
        # Use the normalization stats that were saved with the model
        if device.type == 'cuda':
            ckpt = torch.load(best_path, weights_only=False)
        else:
            ckpt = torch.load(best_path, weights_only=False, map_location=torch.device('cpu'))
        norm = ckpt["norm"]
    else:
        #Fit normalization on training data only
        norm = compute_normalization(
            scalars[idx_train], subints[idx_train], dmcurves[idx_train]
        )

    #Function to split the training, validation, and test sets into scalars, subints, and DM curve values
    #because that's what we use in each branch
    def get_split(idx):
        s, si, dm = apply_normalization(
            scalars[idx], subints[idx], dmcurves[idx], norm
        )
        return s, si, dm, labels[idx]

    train_scalars, train_subints, train_dm, train_labels = get_split(idx_train)
    val_scalars, val_subints, val_dm, val_labels = get_split(idx_val)
    test_scalars, test_subints, test_dm, test_labels  = get_split(idx_test)

    #Create our datasets based on the splits, only augment the training dataset
    train_dataset = PulsarDataset(train_scalars, train_subints, train_dm, train_labels, augment=True)
    val_dataset   = PulsarDataset(val_scalars, val_subints, val_dm, val_labels, augment=False)
    test_dataset  = PulsarDataset(test_scalars, test_subints, test_dm, test_labels, augment=False)

    #DataLoaders for our validation and test sets
    val_loader = DataLoader(
        val_dataset,
        batch_size  = BATCH_SIZE * 2,
        shuffle     = False,
        pin_memory  = True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size  = BATCH_SIZE * 2,
        shuffle     = False,
        pin_memory  = True,
    )

    #Create the model and move to GPU
    model = PulsarNet().to(device)
    #Display the number of parameters to optimize
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters: {n_params:,}")

    #If we are not skipping, train the model
    if not skip_training:
        #Weighted sampler for imbalanced training
        #Weight by the ratio of RFI to pulsar
        n_pos    = int(train_labels.sum())
        n_neg    = len(train_labels) - n_pos
        pos_weight = n_neg / n_pos
        sample_weights = np.where(train_labels == 1, pos_weight, 1.0)
        sampler = WeightedRandomSampler(
            weights = torch.tensor(sample_weights, dtype=torch.float32),
            num_samples = len(train_labels),
            replacement = True,
        )
        print(f"\nSampler: pos_weight = {pos_weight:.1f}x")

        train_loader = DataLoader(
            train_dataset,
            batch_size = BATCH_SIZE,
            sampler = sampler,
            pin_memory = True,
        )

        #We used AdamW optimizer as it is apparently the standard in the industry
        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        #The scheduler adjusts the learning rate based on training
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
        #Our criterion is the Focal Loss function defined above
        criterion = FocalLoss(alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA)
        if device.type == 'cuda':
            scaler = GradScaler('cuda')
        else:
            scaler = GradScaler('cpu')

        print(f"\nTraining for {NUM_EPOCHS} epochs")
        best_ap = 0.0

        #Train for NUM_EPOCHS epochs. We trained for 30.
        for epoch in range(1, NUM_EPOCHS + 1):
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, scaler)
            scheduler.step()

            val_auroc, val_ap, _, _, _ = evaluate(model, val_loader)

            print(
                f"Epoch {epoch:02d}/{NUM_EPOCHS} | "
                f"Loss: {train_loss:.4f} | "
                f"Val AUROC: {val_auroc:.4f} | "
                f"Val AP: {val_ap:.4f}"
            )

            #Save best model by Average Precision (AP).
            if val_ap > best_ap:
                best_ap = val_ap
                torch.save({
                    "epoch"      : epoch,
                    "model_state": model.state_dict(),
                    "val_auroc"  : val_auroc,
                    "val_ap"     : val_ap,
                    "norm"       : norm,
                }, best_path)
                print(f"Saved best model to: {best_path} (AP: {best_ap:.4f})")

    #Load the best model for final evaluation on test.
    print(f"\nTest Set Evaluation")
    if device.type == 'cuda':
        checkpoint = torch.load(best_path, weights_only=False)
    else:
        checkpoint = torch.load(best_path, weights_only=False, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint["model_state"])
    print(f"Loaded model from epoch {checkpoint['epoch']} (Val AP: {checkpoint['val_ap']:.4f})")

    #Tune the threshold on the validation set
    val_auroc, val_ap, val_scores, val_labels_np, _ = evaluate(model, val_loader)
    optimal_threshold = find_optimal_threshold(val_labels_np, val_scores, min_recall=0.95)

    checkpoint["tuned_threshold"] = float(optimal_threshold)
    torch.save(checkpoint, best_path)

    #Evaluate on test with both default and tuned thresholds
    test_auroc, test_ap, test_scores, test_labels_np, test_preds = evaluate(model, test_loader)

    print(f"\nTest AUROC            : {test_auroc:.4f}")
    print(f"Test Average Precision: {test_ap:.4f}")
    print(f"\nClassification Report (default threshold={DEFAULT_THRESHOLD}):")
    print(classification_report(test_labels_np, test_preds, target_names=["RFI", "Pulsar"]))

    tuned_preds = (test_scores >= optimal_threshold).astype(int)
    print(f"\nClassification Report (tuned threshold={optimal_threshold:.4f}):")
    print(classification_report(test_labels_np, tuned_preds, target_names=["RFI", "Pulsar"]))

    #After this point, we want to inspect our classifications to see which files our model struggled with
    print(f"\nInspecting test-set classifications")

    #We reconstruct our json files list, if the length doesn't match we throw an error
    all_json_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".json")]
    if len(all_json_files) != len(labels):
        print(f"  Warning: {len(all_json_files)} files on disk vs {len(labels)} in cache.")
        print(f"  Filename mapping may be off — delete the cache and reprocess to fix.")

    #Map each test sample back to its original filename
    test_filenames    = [all_json_files[orig_idx] for orig_idx in idx_test]
    test_orig_scalars = scalars[idx_test]

    #Use the tuned threshold
    predicted_pulsar = test_scores >= optimal_threshold
    actually_pulsar  = test_labels_np == 1

    #Create an internal function just to print everything in nicely formatted rows
    def _print_row(i):
        fname  = test_filenames[i]
        score  = test_scores[i]
        period = test_orig_scalars[i, 0]
        dm     = test_orig_scalars[i, 2]
        snr    = test_orig_scalars[i, 5]
        print(f"  {fname:<48} score={score:6.4f}  P={period:7.4f}s  DM={dm:6.1f}  S/N={snr:5.1f}")

    #Find the false negatives (the model marked as RFI but it was actually a pulsar)
    fn_indices = np.where((~predicted_pulsar) & actually_pulsar)[0]
    #Sort by score ascending
    fn_sorted = fn_indices[np.argsort(test_scores[fn_indices])]

    print(f"\nMissed pulsars (false negatives): {len(fn_sorted)} total")
    print(f"  Threshold = {optimal_threshold:.4f}")
    if len(fn_sorted) > 0:
        for i in fn_sorted:
            _print_row(i)

    #Find the true positives (correctly identified as pulsar) and display some of the most confident identifications
    tp_indices = np.where(predicted_pulsar & actually_pulsar)[0]
    tp_sorted  = tp_indices[np.argsort(-test_scores[tp_indices])]

    n_show = min(15, len(tp_sorted))
    print(f"\nCorrectly identified pulsars (true positives): {len(tp_sorted)} total")
    print(f"  Top {n_show} by model confidence:")
    for i in tp_sorted[:n_show]:
        _print_row(i)

    #Also display some of the least confident correct pulsar classifications to show which pulsars our model struggled with
    n_borderline = min(10, len(tp_sorted))
    print(f"\nLowest-confidence correct hits (the borderline catches):")
    for i in tp_sorted[-n_borderline:]:
        _print_row(i)

if __name__ == "__main__":
    main()
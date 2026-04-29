import os
import csv
import numpy as np
import torch
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    classification_report,
)

from pulsar_net import (
    PulsarNet,
    preprocess_json_files,
    apply_normalization,
    device,
)


DATA_DIR        = "./S0_json_test_only" 
PULSAR_LIST     = "./known_pulsars.txt"           #set to None if we are classifying unknown candidates
CHECKPOINT_PATH = "./checkpoints/best_model.pt"   #the trained model file
CACHE_PATH      = "./eval_cache.npz"
OUTPUT_CSV      = "./predictions.csv"

BATCH_SIZE       = 128
TARGET_RECALL    = 0.95   #used for tuned threshold
DEFAULT_THRESHOLD = 0.5


if not os.path.exists(CACHE_PATH):
    if PULSAR_LIST is None or not os.path.exists(PULSAR_LIST):
        #preprocess_json_files needs a label file. If we don't have one, write a
        #dummy empty one so every candidate gets label 0 (we'll ignore labels later).
        dummy_label_path = os.path.join(os.path.dirname(CACHE_PATH) or ".", "_dummy_labels.txt")
        with open(dummy_label_path, "w") as f:
            f.write("")
        preprocess_json_files(DATA_DIR, dummy_label_path, CACHE_PATH)
        os.remove(dummy_label_path)
        labels_known = False
    else:
        preprocess_json_files(DATA_DIR, PULSAR_LIST, CACHE_PATH)
        labels_known = True
else:
    print(f"Using existing cache at {CACHE_PATH}")
    labels_known = PULSAR_LIST is not None and os.path.exists(PULSAR_LIST)

print(f"\nLoading dataset from {CACHE_PATH}")

data     = np.load(CACHE_PATH)
scalars  = data["scalars"]
subints  = data["subints"]
dmcurves = data["dmcurves"]
labels   = data["labels"]

print(f"  {len(labels)} samples loaded")

if labels_known:
    print(f"Known pulsars in this set: {int(labels.sum())}")

#Recover the JSON filenames in cache order so we can write them to the CSV
all_json_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".json")]

if len(all_json_files) != len(labels):
    print(f"Warning: {len(all_json_files)} files on disk vs {len(labels)} in cache.")
    print(f"Filename mapping may be off — delete {CACHE_PATH} to rebuild.")


print(f"\nLoading checkpoint from {CHECKPOINT_PATH}")
ckpt = torch.load(CHECKPOINT_PATH, weights_only=False, map_location=device)
norm = ckpt["norm"]
print(f"Epoch: {ckpt['epoch']}  |  AP at training: {ckpt['val_ap']:.4f}")

# Apply normalization (uses the stats saved with the model)
scalars, subints, dmcurves = apply_normalization(scalars, subints, dmcurves, norm)

model = PulsarNet().to(device)
model.load_state_dict(ckpt["model_state"])
model.eval()


print(f"\nRunning inference on {len(labels)} candidates...")
all_scores = []

with torch.no_grad():
    for i in range(0, len(labels), BATCH_SIZE):
        s  = torch.tensor(scalars[i:i + BATCH_SIZE], dtype=torch.float32).to(device)
        si = torch.tensor(subints[i:i + BATCH_SIZE], dtype=torch.float32).to(device)
        dm = torch.tensor(dmcurves[i:i + BATCH_SIZE], dtype=torch.float32).to(device)

        logits = model(s, si, dm)
        all_scores.append(torch.sigmoid(logits).cpu().numpy())

scores = np.concatenate(all_scores)

tuned_threshold = ckpt.get("tuned_threshold", DEFAULT_THRESHOLD)
print(f"\nUsing tuned threshold from checkpoint: {tuned_threshold:.4f}")

#Calculate the metrics only if we have a list of known pulsars
if labels_known and labels.sum() > 0:
    print(f"\n=== Metrics ===")
    print(f"AUROC            : {roc_auc_score(labels, scores):.4f}")
    print(f"Average Precision: {average_precision_score(labels, scores):.4f}")

    print(f"\nClassification Report (default threshold={DEFAULT_THRESHOLD}):")
    preds_default = (scores >= DEFAULT_THRESHOLD).astype(int)
    print(classification_report(labels, preds_default, target_names=["RFI", "Pulsar"]))

    print(f"Classification Report (tuned threshold={tuned_threshold:.4f}):")
    preds_tuned = (scores >= tuned_threshold).astype(int)
    print(classification_report(labels, preds_tuned, target_names=["RFI", "Pulsar"]))
else:
    print("\nNo labels available, skipping metrics.")

#Write the predictions to a CSV file
print(f"\nWriting predictions to {OUTPUT_CSV}")
with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    header = [
        "filename",
        "score",
        f"predicted_pulsar_at_{tuned_threshold:.4f}",
        "predicted_pulsar_at_0.5",
    ]
    if labels_known:
        header.append("true_label")
    writer.writerow(header)

    for i in range(len(scores)):
        fname = all_json_files[i] if i < len(all_json_files) else f"index_{i}"
        row = [
            fname,
            f"{scores[i]:.6f}",
            int(scores[i] >= tuned_threshold),
            int(scores[i] >= DEFAULT_THRESHOLD),
        ]
        if labels_known:
            row.append(int(labels[i]))
        writer.writerow(row)

print(f"Done. {len(scores)} predictions written.")
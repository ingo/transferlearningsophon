import os
import argparse
import sys
import csv
import math
import torch
import uproot
import numpy as np
from tqdm import tqdm
from math import cos, sin, sinh

parser = argparse.ArgumentParser(description="JetClass inference with optional debug output.")
parser.add_argument('--input-dir', type=str, default="./data/JetClass/val_5M", help='Directory containing input ROOT files.')
parser.add_argument('--output-csv', type=str, default="HToCC_inference_with_embedding.csv", help='Path for output CSV file.')
parser.add_argument('--files', nargs='+', default=["HToCC_120.root", "HToCC_121.root", "HToCC_122.root", "HToCC_123.root", "HToCC_124.root"], help='List of ROOT files to process.')
parser.add_argument('--debug', action='store_true', help='Enable debug print statements for feature shapes.')
parser.add_argument('--log-file', type=str, default=None, help='Optional path to write error/debug output log.')
args = parser.parse_args()

######################################################################
# Ensure the current directory is in sys.path to import local modules
# --- Debug flag for optional verbose output ---
# Allows you to run the script with --debug to enable extra print statements for troubleshooting.
######################################################################
sys.path.append(".")
# Import the get_model function from the local networks module
from networks.example_ParticleTransformer_sophon import get_model

# Define the list of particle and scalar feature keys to read from the ROOT files
particle_keys = [
    'part_px', 'part_py', 'part_pz', 'part_energy',
    'part_deta', 'part_dphi', 'part_d0val', 'part_d0err',
    'part_dzval', 'part_dzerr', 'part_charge',
    'part_isChargedHadron', 'part_isNeutralHadron',
    'part_isPhoton', 'part_isElectron', 'part_isMuon'
]

# Define scalar features that have a single value per jet
scalar_keys = [
    'label_QCD','label_Hbb','label_Hcc','label_Hgg',
    'label_H4q','label_Hqql','label_Zqq','label_Wqq',
    'label_Tbqq','label_Tbl','jet_pt','jet_eta','jet_phi',
    'jet_energy','jet_nparticles','jet_sdmass','jet_tau1',
    'jet_tau2','jet_tau3','jet_tau4','aux_genpart_eta',
    'aux_genpart_phi','aux_genpart_pid','aux_genpart_pt',
    'aux_truth_match'
]

pf_keys = particle_keys + scalar_keys

root_dir = args.input_dir
root_files = args.files
output_csv_path = args.output_csv

# Container to hold configuration settings for get_model function
class DummyDataConfig:
    # map of input feature indices
    input_dicts = {"pf_features": list(range(37))}
    # Name of the model input
    input_names = ["pf_points"]
    # Expected tensor shape for the model input
    input_shapes = {"pf_points": (128, 37)}
    # Names of output labels
    label_names = ["label"]
    # Number of output classes
    num_classes = 10


data_config = DummyDataConfig()

# Check for GPU availability and load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))

# Create the model
model, _ = get_model(data_config, num_classes=data_config.num_classes, export_embed=True)
# Put the model in evaluation mode and move it to the appropriate device
model.eval().to(device)



######################################################################
# Optionally open a log file for error/debug output
log_fh = open(args.log_file, "w") if args.log_file else None
def logprint(*a, **k):
    print(*a, **k)
    if log_fh:
        print(*a, **k, file=log_fh)

# Open the CSV file in write mode ('w')
with open(output_csv_path, mode="w", newline="") as csvfile:
    # Create a CSV writer object
    writer = csv.writer(csvfile)

    # Define the base header for the CSV file (general jet info and labels)
    base_header = (["file", "event_index"] +
                   ["truth_label", "label_name",
                    "jet_sdmass", "jet_mass", "jet_pt", "jet_eta", "jet_phi"])
    # Create column headers for the 128-dimensional output embedding vector
    emb_header = [f"emb_{j}" for j in range(128)]
    # Write the complete header row to the CSV file
    writer.writerow(base_header + emb_header)

    # Iterate over each ROOT file in the list
    for file_name in root_files:
        print(f"\nRunning inference on: {file_name}")
        # Construct the full file path
        file_path = os.path.join(root_dir, file_name)
        # Open the ROOT file using uproot
        with uproot.open(file_path) as f:
            # Access the 'tree' object (contains the data)
            tree = f["tree"]
            # Read the required feature arrays from the tree into a NumPy dictionary
            arrays = tree.arrays(pf_keys, library="np")
        # Validation: check for missing keys in arrays
        missing_keys = [k for k in pf_keys if k not in arrays]
        if missing_keys:
            logprint(f"[ERROR] Missing keys in file {file_name}: {missing_keys}. Skipping file.")
            continue
        # Define the maximum number of particles the model can handle (128)
        max_part = 128
        # Get the total number of events (jets) in the file
        total_events = len(arrays["part_px"])
        # Validation: check all arrays have the same number of events
        for k in pf_keys:
            if len(arrays[k]) != total_events:
                logprint(f"[ERROR] Key '{k}' in file {file_name} has {len(arrays[k])} events, expected {total_events}. Skipping file.")
                continue

        # Loop through each event in the file, showing a progress bar
        error_count = 0
        processed_count = 0
        for i in tqdm(range(total_events), desc=f"{file_name}"):
            try:
                # Get the actual number of constituent particles in the current jet
                n_part = arrays["part_px"][i].shape[0]
                # --- Validation and sanity checks ---
                # Check that all expected keys are present and have correct shape
                for k in particle_keys:
                    if k not in arrays:
                        logprint(f"[ERROR] Missing particle key '{k}' in file {file_name}")
                        raise KeyError(f"Missing particle key '{k}' in file {file_name}")
                    if arrays[k][i].shape[0] != n_part:
                        logprint(f"[ERROR] Particle key '{k}' has wrong shape in event {i} of {file_name}")
                        raise ValueError(f"Particle key '{k}' has wrong shape in event {i} of {file_name}")
                for k in scalar_keys:
                    if k not in arrays:
                        logprint(f"[ERROR] Missing scalar key '{k}' in file {file_name}")
                        raise KeyError(f"Missing scalar key '{k}' in file {file_name}")
                # Skip the event if it has more particles than the model's limit
                if n_part > max_part:
                    continue

                # build input tensor
                # Extract particle features for the current event
                particle_feats = [arrays[k][i] for k in particle_keys]
                # --- Improved scalar feature handling ---
                # Only tile if the value is a scalar, otherwise use the array directly.
                # This ensures that only true scalars are broadcasted to match the number of particles.
                #
                # Example:
                #   Scalar: arrays[k][i] = 42.0 (float or int)
                #   After np.full(n_part, val): array([42., 42., ..., 42.]) shape=(n_part,)
                #   Array:  arrays[k][i] = array([1.1, 2.2, 3.3, ...]) shape=(n_part,)
                #
                # If a feature is already an array of the correct shape, it is used as-is.
                # If the shape is unexpected, it prints debug info and raises an error.
                scalar_feats = []
                for k in scalar_keys:
                    val = arrays[k][i]
                    if np.isscalar(val) or (isinstance(val, np.ndarray) and val.shape == ()):  # true scalar
                        scalar_feats.append(np.full(n_part, val))
                    elif isinstance(val, np.ndarray) and val.shape == (n_part,):
                        scalar_feats.append(val)
                    else:
                        # If the shape is not as expected, print debug info and raise
                        print(f"Event {i}, Feature {k}: unexpected shape {getattr(val, 'shape', 'scalar')}")
                        raise ValueError(f"Unexpected shape for feature {k} in event {i}: {getattr(val, 'shape', 'scalar')}")
                # --- Conditional debug printing for features ---
                # This prints the type, shape, and first few values of each feature for the current event, but only if --debug is enabled.
                if args.debug:
                    for idx, (k, arr) in enumerate(zip(particle_keys + scalar_keys, particle_feats + scalar_feats)):
                        print(f"Event {i}, Feature {k}: type={type(arr)}, shape={getattr(arr, 'shape', 'scalar')}, first5={arr[:5] if hasattr(arr, '__getitem__') else arr}")
                # Combine particle and scalar features
                all_feats = particle_feats + scalar_feats
                # Stack all features into a 2D array (n_part, n_features)
                pf_features = np.stack(all_feats, axis=1)

                # Create a zero-padded array for the model input (max_part, n_features)
                padded = np.zeros((max_part, pf_features.shape[1]), dtype=np.float32)
                # Fill the array with the actual feature data
                padded[:n_part, :] = pf_features
                
                # Convert the NumPy array to a PyTorch tensor, add a batch dimension (unsqueeze(0)), and move it to the device
                jet_tensor = torch.tensor(padded, dtype=torch.float32).unsqueeze(0).to(device)
                # Extract Lorentz vectors (px, py, pz, E) and permute dimensions (batch, feature, particle)
                lorentz_vectors = jet_tensor[:, :, 0:4].transpose(1, 2)
                # Extract non-kinematic features and permute dimensions (batch, feature, particle)
                features = jet_tensor[:, :, 4:].transpose(1, 2)
                # Create a mask tensor: 1 for valid particles, 0 for padded (checks if the sum of all features for a particle is non-zero)
                mask = (jet_tensor.sum(dim=2) != 0).unsqueeze(1)
                # Placeholder for particle coordinates if needed (not used here)
                points = None

                # Disable gradient calculation during inference to save memory and speed up computation
                with torch.no_grad():
                    # --- Model output handling (tuple or single) ---
                    # This allows the code to work whether the model returns a tuple (logits, embedding) or just a single output (embedding).
                    #
                    # Example model outputs:
                    #   Tuple: (logits, embedding)
                    #     logits:    torch.Tensor of shape (1, num_classes)
                    #     embedding: torch.Tensor of shape (1, 128)
                    #   Single: embedding only
                    #     embedding: torch.Tensor of shape (1, 128)
                    #
                    # If only one value is returned, it is assumed to be the embedding. If --debug is enabled, the output type and value are printed.
                    model_output = model(points, features, lorentz_vectors, mask)
                    if isinstance(model_output, tuple) and len(model_output) == 2:
                        logits, embedding = model_output
                    else:
                        if args.debug:
                            print(f"Model output type: {type(model_output)}, value: {model_output}")
                        # Try to handle single output as embedding
                        embedding = model_output
                        logits = None
                    # Remove the batch dimension, move the tensor to CPU, and convert it to a NumPy array
                    embedding = embedding.squeeze(0).cpu().numpy()

                # truth labels
                # Extract the one-hot encoded truth labels for the current jet
                label_array = np.array([arrays[k][i] for k in [
                    'label_QCD', 'label_Hbb', 'label_Hcc', 'label_Hgg',
                    'label_H4q', 'label_Hqql', 'label_Zqq', 'label_Wqq',
                    'label_Tbqq', 'label_Tbl'
                ]])
                # Determine the true class index (0-9) by finding the index of the maximum value (1)
                truth_label = int(np.argmax(label_array))
                # List of label names
                label_names = ["QCD","Hbb","Hcc","Hgg","H4q","Hqql","Zqq","Wqq","Tbqq","Tbl"]
                # Get the corresponding name for the true class
                label_name = label_names[truth_label]

                # softdrop + ungroomed mass
                # Get the SoftDrop mass value
                jet_sdmass = float(arrays["jet_sdmass"][i])
                # Get the jet's kinematic properties
                pt  = float(arrays["jet_pt"][i])
                eta = float(arrays["jet_eta"][i])
                phi = float(arrays["jet_phi"][i])
                E   = float(arrays["jet_energy"][i])

                # Convert jet's transverse momentum (pt), pseudorapidity (eta), and azimuthal angle (phi) to Cartesian momentum components (px, py, pz)
                px = pt * cos(phi)
                py = pt * sin(phi)
                # The relationship between pz, pt, and eta is: pz = pt * sinh(eta)
                pz = pt * sinh(eta)
                # Calculate the square of the three-momentum (p^2 = px^2 + py^2 + pz^2)
                p2 = px*px + py*py + pz*pz
                # Calculate the square of the invariant mass (m^2 = E^2 - p^2). Ensure it's non-negative.
                m2 = max(E*E - p2, 0.0)
                # Calculate the invariant mass (ungroomed jet mass)
                jet_mass = float(np.sqrt(m2))

                # Create the data row for the CSV file: file info, labels, jet properties, and the embedding
                row = [file_name, i, truth_label, label_name,
                       jet_sdmass, jet_mass, pt, eta, phi] + list(embedding)
                writer.writerow(row)
                processed_count += 1

            # Handle any exceptions (errors) that occur during the processing of a specific event
            except Exception as e:
                # --- Enhanced exception debugging ---
                # If an error occurs, the script prints the error and, if debugging is enabled, prints detailed feature info and the full traceback for easier troubleshooting.
                import traceback
                logprint(f"Error in event {i}: {e}")
                if args.debug:
                    logprint("Feature debug info for failed event:")
                    for idx, (k, arr) in enumerate(zip(particle_keys + scalar_keys, particle_feats + scalar_feats)):
                        logprint(f"  Feature {k}: type={type(arr)}, shape={getattr(arr, 'shape', 'scalar')}, first5={arr[:5] if hasattr(arr, '__getitem__') else arr}")
                    traceback.print_exc(file=log_fh if log_fh else None)
                # Continue to the next event if an error occurs
                error_count += 1
                continue

######################################################################
# Print summary and close log file if used
print("\n==================== SUMMARY ====================")
print(f"Processed files: {len(root_files)}")
print(f"Total events processed: {processed_count}")
print(f"Total errors/skipped events: {error_count}")
print(f"Saved CSV data to {output_csv_path}")
if log_fh:
    print(f"Log written to {args.log_file}")
    log_fh.close()
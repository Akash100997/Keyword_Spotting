import os
import numpy as np
import librosa as lb

DATASET_PATH = "Dataset"  # The path to the dataset
NPZ_DATA_FILE = "data.npz"
SAMPLES_TO_CONSIDER = 16000


def prepare_dataset(dataset_path, n_mfcc=13, hop_length=512, n_fft=2048):
    # Data Dictionary
    data = {
        "mappings": [],
        "labels": [],
        "MFCCs": [],
        "files": []
    }

    # Loop through all the sub directories
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):

        # Ensure that it doesn't recurse to the root level
        if dirpath is not dataset_path:

            # Update the mappings
            category = dirpath.split("\\")[-1]  # Dataset/down => down
            data["mappings"].append(category)
            print(f"Processing {category}")

            # Loop through all the files and extract the features
            for f in filenames:
                # Get the file path
                file_path = os.path.join(dirpath, f)

                # Load the audio file
                signal, sr = lb.load(file_path, sr=None)

                # Ensure the data is 1 second long
                if len(signal) >= SAMPLES_TO_CONSIDER:
                    signal = signal[:SAMPLES_TO_CONSIDER]

                    # Extracting MFCC's
                    MFCCs = lb.feature.mfcc(signal, n_mfcc=n_mfcc, hop_length=hop_length, n_fft=n_fft)
                    # Store all the retrieved data to the dictionary
                    data["labels"].append(i - 1)
                    data["MFCCs"].append(MFCCs.T.tolist())
                    data["files"].append(file_path)
                    print(f"{file_path}:{i - 1}")

    np.savez(NPZ_DATA_FILE,
             mappings = data["mappings"],
             labels = data["labels"],
             MFCCs = data["MFCCs"],
             files = data["files"]             
             )


if __name__ == "__main__":
    prepare_dataset(DATASET_PATH)

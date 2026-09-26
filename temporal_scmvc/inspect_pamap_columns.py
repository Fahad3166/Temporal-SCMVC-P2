import os
import numpy as np

data_path = '/Users/fahad/Documents/MY#Documents/FAHAD ALI/Uni Wien/6th Semester/P-1/SCMVC/data/pamap2+physical+activity+monitoring/PAMAP2_Dataset/Protocol'

files = sorted([f for f in os.listdir(data_path) if f.endswith(".dat")])
print("Found files:", files)

first_file = os.path.join(data_path, files[0])
data = np.loadtxt(first_file)

print("Raw file shape:", data.shape)

# remove activity 0 rows
data = data[data[:, 1] != 0]
print("After removing activity 0:", data.shape)

# labels
y = data[:, 1].astype(int)
print("Unique labels:", np.unique(y))

# features after removing timestamp and activity
X = data[:, 2:]
print("Feature matrix shape after removing first 2 cols:", X.shape)

# print column count clearly
print("Number of feature columns:", X.shape[1])
import numpy as np

# Load train data
X_train = np.loadtxt('/Users/fahad/Documents/MY#Documents/FAHAD ALI/Uni Wien/6th Semester/P-1/SCMVC/HAR_dataset/UCI HAR Dataset/train/X_train.txt')
y_train = np.loadtxt('/Users/fahad/Documents/MY#Documents/FAHAD ALI/Uni Wien/6th Semester/P-1/SCMVC/HAR_dataset/UCI HAR Dataset/train/y_train.txt')

print("X shape:", X_train.shape)
print("Y shape:", y_train.shape)

# Split into 4 views
num_views = 4
split_size = X_train.shape[1] // num_views

views = []

for i in range(num_views):
    start = i * split_size
    end = (i + 1) * split_size if i != num_views - 1 else X_train.shape[1]
    view = X_train[:, start:end]
    views.append(view)
    print(f"View {i+1} shape:", view.shape)

print("Total views:", len(views))
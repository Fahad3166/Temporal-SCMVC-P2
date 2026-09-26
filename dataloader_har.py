from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
from torch.utils.data import Dataset
import scipy.io
import torch


class BDGP(Dataset):
    def __init__(self, path):
        data1 = scipy.io.loadmat(path+'BDGP.mat')['X1'].astype(np.float32)
        data2 = scipy.io.loadmat(path+'BDGP.mat')['X2'].astype(np.float32)
        labels = scipy.io.loadmat(path+'BDGP.mat')['Y'].transpose()
        self.x1 = data1
        self.x2 = data2
        self.y = labels

    def __len__(self):
        return self.x1.shape[0]

    def __getitem__(self, idx):
        return [torch.from_numpy(self.x1[idx]), torch.from_numpy(
           self.x2[idx])], torch.from_numpy(self.y[idx]), torch.from_numpy(np.array(idx)).long()


class CCV(Dataset):
    def __init__(self, path):
        self.data1 = np.load(path+'STIP.npy').astype(np.float32)
        scaler = MinMaxScaler()
        self.data1 = scaler.fit_transform(self.data1)
        self.data2 = np.load(path+'SIFT.npy').astype(np.float32)
        self.data3 = np.load(path+'MFCC.npy').astype(np.float32)
        self.labels = np.load(path+'label.npy')

    def __len__(self):
        return 6773

    def __getitem__(self, idx):
        x1 = self.data1[idx]
        x2 = self.data2[idx]
        x3 = self.data3[idx]
        return [torch.from_numpy(x1), torch.from_numpy(
           x2), torch.from_numpy(x3)], torch.from_numpy(self.labels[idx]), torch.from_numpy(np.array(idx)).long()


class MNIST_USPS(Dataset):
    def __init__(self, path):
        self.Y = scipy.io.loadmat(path + 'MNIST_USPS.mat')['Y'].astype(np.int32).reshape(5000,)
        self.V1 = scipy.io.loadmat(path + 'MNIST_USPS.mat')['X1'].astype(np.float32)
        self.V2 = scipy.io.loadmat(path + 'MNIST_USPS.mat')['X2'].astype(np.float32)

    def __len__(self):
        return 5000

    def __getitem__(self, idx):
        x1 = self.V1[idx].reshape(784)
        x2 = self.V2[idx].reshape(784)
        return [torch.from_numpy(x1), torch.from_numpy(x2)], self.Y[idx], torch.from_numpy(np.array(idx)).long()


class Fashion(Dataset):
    def __init__(self, path):
        self.Y = scipy.io.loadmat(path + 'Fashion.mat')['Y'].astype(np.int32).reshape(10000,)
        self.V1 = scipy.io.loadmat(path + 'Fashion.mat')['X1'].astype(np.float32)
        self.V2 = scipy.io.loadmat(path + 'Fashion.mat')['X2'].astype(np.float32)
        self.V3 = scipy.io.loadmat(path + 'Fashion.mat')['X3'].astype(np.float32)

    def __len__(self):
        return 10000

    def __getitem__(self, idx):
        x1 = self.V1[idx].reshape(784)
        x2 = self.V2[idx].reshape(784)
        x3 = self.V3[idx].reshape(784)
        return [torch.from_numpy(x1), torch.from_numpy(x2), torch.from_numpy(x3)], self.Y[idx], torch.from_numpy(np.array(idx)).long()


class Caltech(Dataset):
    def __init__(self, path, view):
        data = scipy.io.loadmat(path)
        scaler = MinMaxScaler()
        self.view1 = scaler.fit_transform(data['X1'].astype(np.float32))
        self.view2 = scaler.fit_transform(data['X2'].astype(np.float32))
        self.view3 = scaler.fit_transform(data['X3'].astype(np.float32))
        self.view4 = scaler.fit_transform(data['X4'].astype(np.float32))
        self.view5 = scaler.fit_transform(data['X5'].astype(np.float32))
        self.labels = scipy.io.loadmat(path)['Y'].transpose()
        self.view = view

    def __len__(self):
        return 1400

    def __getitem__(self, idx):
        if self.view == 2:
            return [torch.from_numpy(self.view1[idx]), torch.from_numpy(self.view2[idx])], \
                   torch.from_numpy(self.labels[idx]), torch.from_numpy(np.array(idx)).long()
        if self.view == 3:
            return [torch.from_numpy(self.view1[idx]), torch.from_numpy(self.view2[idx]),
                    torch.from_numpy(self.view5[idx])], \
                   torch.from_numpy(self.labels[idx]), torch.from_numpy(np.array(idx)).long()
        if self.view == 4:
            return [torch.from_numpy(self.view1[idx]), torch.from_numpy(self.view2[idx]),
                    torch.from_numpy(self.view5[idx]), torch.from_numpy(self.view4[idx])], \
                   torch.from_numpy(self.labels[idx]), torch.from_numpy(np.array(idx)).long()
        if self.view == 5:
            return [torch.from_numpy(self.view1[idx]), torch.from_numpy(self.view2[idx]),
                    torch.from_numpy(self.view5[idx]), torch.from_numpy(self.view4[idx]),
                    torch.from_numpy(self.view3[idx])], \
                   torch.from_numpy(self.labels[idx]), torch.from_numpy(np.array(idx)).long()


class cifar_10(Dataset):
    def __init__(self, path):
        data = scipy.io.loadmat(path + 'cifar10.mat')
        self.Y = data['truelabel'][0][0].astype(np.int32).reshape(50000,)
        self.V1 = data['data'][0][0].T.astype(np.float32)
        self.V2 = data['data'][1][0].T.astype(np.float32)
        self.V3 = data['data'][2][0].T.astype(np.float32)

    def __len__(self):
        return 50000

    def __getitem__(self, idx):
        return [torch.from_numpy(self.V1[idx]), torch.from_numpy(self.V2[idx]),
                torch.from_numpy(self.V3[idx])], self.Y[idx], torch.from_numpy(np.array(idx)).long()


class cifar_100(Dataset):
    def __init__(self, path):
        data = scipy.io.loadmat(path + 'cifar100.mat')
        self.Y = data['truelabel'][0][0].astype(np.int32).reshape(50000,)
        self.V1 = data['data'][0][0].T.astype(np.float32)
        self.V2 = data['data'][1][0].T.astype(np.float32)
        self.V3 = data['data'][2][0].T.astype(np.float32)

    def __len__(self):
        return 50000

    def __getitem__(self, idx):
        return [torch.from_numpy(self.V1[idx]), torch.from_numpy(self.V2[idx]),
                torch.from_numpy(self.V3[idx])], self.Y[idx], torch.from_numpy(np.array(idx)).long()


class synthetic3d(Dataset):
    def __init__(self, path):
        data = scipy.io.loadmat(path + 'synthetic3d.mat')
        self.Y = data['Y'].astype(np.int32).reshape(600,)
        self.V1 = data['X'][0][0].astype(np.float32)
        self.V2 = data['X'][1][0].astype(np.float32)
        self.V3 = data['X'][2][0].astype(np.float32)

    def __len__(self):
        return 600

    def __getitem__(self, idx):
        return [torch.from_numpy(self.V1[idx]), torch.from_numpy(self.V2[idx]),
                torch.from_numpy(self.V3[idx])], self.Y[idx], torch.from_numpy(np.array(idx)).long()


class prokaryotic(Dataset):
    def __init__(self, path):
        data = scipy.io.loadmat(path + 'prokaryotic.mat')
        self.Y = data['Y'].astype(np.int32).reshape(551,)
        self.V1 = data['X'][0][0].astype(np.float32)
        self.V2 = data['X'][1][0].astype(np.float32)
        self.V3 = data['X'][2][0].astype(np.float32)

    def __len__(self):
        return 551

    def __getitem__(self, idx):
        return [torch.from_numpy(self.V1[idx]), torch.from_numpy(self.V2[idx]),
                torch.from_numpy(self.V3[idx])], self.Y[idx], torch.from_numpy(np.array(idx)).long()


# ============================================================
# HAR Dataset — fixed version
# ============================================================
class HAR(Dataset):
    def __init__(self, data_path, num_views=2, split='train'):
        """
        UCI HAR dataset loader with configurable number of views.

        Args:
            data_path : path to the 'UCI HAR Dataset/' folder
            num_views : how many views to split the 561 features into (2-7)
            split     : 'train' or 'test'

        HAR has 561 features. Each view gets an equal-sized slice.
        Leftover features (561 % num_views) are distributed one-by-one
        to the first views so NO features are ever dropped.

        Labels are 1-6 in the raw file → converted to 0-5 for clustering.
        """
        X = np.loadtxt(data_path + f'{split}/X_{split}.txt')
        y = np.loadtxt(data_path + f'{split}/y_{split}.txt')

        print(f"Loaded HAR {split}: {X.shape}, unique labels: {np.unique(y.astype(int))}")

        # Standardise across features
        scaler = StandardScaler()
        X = scaler.fit_transform(X).astype(np.float32)

        # Split into num_views — distribute remainder so nothing is dropped
        total_features = X.shape[1]          # 561
        base = total_features // num_views   # features per view (base)
        remainder = total_features % num_views  # leftover features

        self.views = []
        self.dims = []
        start = 0
        for i in range(num_views):
            # First `remainder` views get one extra feature
            size = base + (1 if i < remainder else 0)
            end = start + size
            view_data = X[:, start:end]
            self.views.append(view_data)
            self.dims.append(size)
            print(f"  View {i+1}: cols {start}–{end-1}  dim={size}")
            start = end

        # FIX: HAR labels are 1-6, convert to 0-5 for metric functions
        self.y = y.astype(int) - 1
        self.num_views = num_views

    def __len__(self):
        return self.views[0].shape[0]

    def __getitem__(self, idx):
        data = [torch.from_numpy(v[idx]).float() for v in self.views]
        label = torch.tensor(self.y[idx]).long()
        return data, label, torch.tensor(idx).long()


# ============================================================
# load_data — central entry point
# ============================================================
def load_data(dataset, har_path=None, har_views=2):
    """
    Returns: (dataset_obj, dims, view, data_size, class_num)

    For HAR:
        har_path  = path to 'UCI HAR Dataset/' folder
        har_views = number of views to create (2, 3, 4, 5, 6, 7)
    """
    if dataset == "BDGP":
        dataset = BDGP('./data/')
        dims = [1750, 79]
        view = 2
        data_size = 2500
        class_num = 5

    elif dataset == "MNIST-USPS":
        dataset = MNIST_USPS('./data/')
        dims = [784, 784]
        view = 2
        class_num = 10
        data_size = 5000

    elif dataset == "CCV":
        dataset = CCV('./data/')
        dims = [5000, 5000, 4000]
        view = 3
        data_size = 6773
        class_num = 20

    elif dataset == "Fashion":
        dataset = Fashion('./data/')
        dims = [784, 784, 784]
        view = 3
        data_size = 10000
        class_num = 10

    elif dataset == "Caltech-2V":
        dataset = Caltech('data/Caltech-5V.mat', view=2)
        dims = [40, 254]
        view = 2
        data_size = 1400
        class_num = 7

    elif dataset == "Caltech-3V":
        dataset = Caltech('data/Caltech-5V.mat', view=3)
        dims = [40, 254, 928]
        view = 3
        data_size = 1400
        class_num = 7

    elif dataset == "Caltech-4V":
        dataset = Caltech('data/Caltech-5V.mat', view=4)
        dims = [40, 254, 928, 512]
        view = 4
        data_size = 1400
        class_num = 7

    elif dataset == "Caltech-5V":
        dataset = Caltech('data/Caltech-5V.mat', view=5)
        dims = [40, 254, 928, 512, 1984]
        view = 5
        data_size = 1400
        class_num = 7

    elif dataset == "Synthetic3d":
        dataset = synthetic3d('./data/')
        dims = [3, 3, 3]
        view = 3
        data_size = 600
        class_num = 3

    elif dataset == "Prokaryotic":
        dataset = prokaryotic('./data/')
        dims = [438, 3, 393]
        view = 3
        data_size = 551
        class_num = 4

    elif dataset == "Cifar10":
        dataset = cifar_10('./data/')
        dims = [512, 2048, 1024]
        view = 3
        data_size = 50000
        class_num = 10

    elif dataset == "Cifar100":
        dataset = cifar_100('./data/')
        dims = [512, 2048, 1024]
        view = 3
        data_size = 50000
        class_num = 100

    elif dataset == "HAR":
        if har_path is None:
            raise ValueError("HAR requires har_path= argument pointing to 'UCI HAR Dataset/' folder")
        dataset = HAR(data_path=har_path, num_views=har_views, split='train')
        dims = dataset.dims          # computed dynamically based on num_views
        view = har_views
        data_size = len(dataset)     # 7352 for train split
        class_num = 6

    else:
        raise NotImplementedError(f"Dataset '{dataset}' not recognised")

    return dataset, dims, view, data_size, class_num
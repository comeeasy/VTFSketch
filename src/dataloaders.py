from torch.utils.data import Dataset, DataLoader

from src.preprocesses import (
    TargetPreprocessor, VTFPreprocessor, 
    ImagePreprocessor, VTFPreprocessorUNet,
    TargetPreprocessorUNet
)



def load_data_dict_from_yaml(yaml_path):
    import yaml
    # Load the YAML file
    with open(yaml_path, 'r') as file:
        data_dict = yaml.safe_load(file)
    
    return data_dict

class FPathDataset(Dataset):
    def __init__(self, config_path) -> None:
        super().__init__()
        self.data = load_data_dict_from_yaml(config_path)

        _len = len(self.data)
        self.vtfs    = [VTFPreprocessor.get(self.data[idx]['vtf']) for idx in range(_len)]
        self.targets = [TargetPreprocessor.get(self.data[idx]['target']) for idx in range(_len)]
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.vtfs[index], self.targets[index]

def get_FPathDatasets(args):
    train_dset = FPathDataset(args.train_yaml)
    valid_dset = FPathDataset(args.val_yaml)
    test_dset = FPathDataset(args.test_yaml)
    
    return train_dset, valid_dset, test_dset

class UNetFPathDataset(Dataset):
    def __init__(self, config_path) -> None:
        super().__init__()
        self.data = load_data_dict_from_yaml(config_path)

        _len = len(self.data)
        self.vtfs    = [VTFPreprocessorUNet.get(self.data[idx]['vtf']) for idx in range(_len)]
        self.targets = [TargetPreprocessorUNet.get(self.data[idx]['target']) for idx in range(_len)]
        self.imgs    = [ImagePreprocessor.get(self.data[idx]['img']) for idx in range(_len)]
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.vtfs[index], self.imgs[index], self.targets[index]

def get_FPathUNetDataset(args):
    train_dset = UNetFPathDataset(args.train_yaml)
    valid_dset = UNetFPathDataset(args.val_yaml)
    test_dset = UNetFPathDataset(args.test_yaml)
    
    return train_dset, valid_dset, test_dset



def get_data_loaders(args, mode="FPathDataset"):
    if   mode == "FPathDataset":
        train_dset, valid_dset, test_dset = get_FPathDatasets(args)
    elif mode == "UNetFPathPredictor":
        train_dset, valid_dset, test_dset = get_FPathUNetDataset(args)
    else:
        raise RuntimeError("model_name must be [\"FPathDataset\" or \"UNetFPathPredictor\"]")
    
    train_loader = DataLoader(
        train_dset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=args.num_workers,
        drop_last=True,
    )
    val_loader = DataLoader(
        valid_dset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    
    return train_loader, val_loader, test_loader
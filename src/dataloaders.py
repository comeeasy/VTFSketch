from torch.utils.data import Dataset, DataLoader

import lightning as L

from src.preprocesses import (
    TargetPreprocessor, 
    VTFPreprocessor, 
    ImagePreprocessor,
    InfodrawPreprocessor,
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
        self.vtfs       = [VTFPreprocessor.get(self.data[idx]['vtf']) for idx in range(_len)]
        self.targets    = [TargetPreprocessor.get(self.data[idx]['target']) for idx in range(_len)]
        self.imgs       = [ImagePreprocessor.get(self.data[idx]['img']) for idx in range(_len)]
        self.infodraws  = [InfodrawPreprocessor.get(self.data[idx]['infodraw'] for idx in range(_len))]
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.vtfs[index], self.imgs[index], self.infodraws[index], self.targets[index]

# class UNetFPathDataset(Dataset):
#     def __init__(self, config_path) -> None:
#         super().__init__()
#         self.data = load_data_dict_from_yaml(config_path)

#         _len = len(self.data)
#         self.vtfs    = [VTFPreprocessor.get(self.data[idx]['vtf']) for idx in range(_len)]
#         self.targets = [TargetPreprocessor.get(self.data[idx]['target']) for idx in range(_len)]
#         self.imgs    = [ImagePreprocessor.get(self.data[idx]['img']) for idx in range(_len)]
    
#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, index):
#         return self.vtfs[index], self.imgs[index], self.targets[index]
    
class FPathLazyDataset(Dataset):
    def __init__(self, config_path) -> None:
        super().__init__()
        self.data = load_data_dict_from_yaml(config_path)
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        vtf_path        = self.data[index]['vtf']
        img_path        = self.data[index]['img']
        target_path     = self.data[index]['target']
        infodraw_path   = self.data[index]['infodraw']
        
        vtf         = VTFPreprocessor.get(vtf_path=vtf_path)
        img         = ImagePreprocessor.get(img_path=img_path)
        target      = TargetPreprocessor.get(target_path=target_path)
        infodraw    = InfodrawPreprocessor.get(infodraw_path=infodraw_path)
        
        return vtf, img, infodraw, target


class FPathDataModule(L.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args

    def setup(self, stage=None):
        if self.args.use_lazy_loader:
            self.train_dataset = FPathLazyDataset(self.args.train_yaml)
            self.val_dataset = FPathLazyDataset(self.args.val_yaml)
            self.test_dataset = FPathLazyDataset(self.args.test_yaml)
        else:
            self.train_dataset = FPathDataset(self.args.train_yaml)
            self.val_dataset = FPathDataset(self.args.val_yaml)
            self.test_dataset = FPathDataset(self.args.test_yaml)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=self.args.num_workers,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=self.args.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=self.args.num_workers,
        )
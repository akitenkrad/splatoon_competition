import unittest
from pathlib import Path
import torch
from ..datasets import SplatoonDataset

class TestSplatoonDataset(unittest.TestCase):
    
    data_path = 'dataset/dev_data.csv'
    
    def test_dataloader(self):
        dataset = SplatoonDataset(self.data_path)
        
        data_count = 0
        for _ in torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True):
            data_count += 1
        self.assertTrue(data_count == len(dataset))
        
    def test_train_test_split(self):
        dataset = SplatoonDataset(self.data_path)
        train_ds, test_ds = dataset.train_test_split(test_size=0.1)
        
        data_count = 0
        for _ in torch.utils.data.DataLoader(train_ds, batch_size=1, shuffle=True):
            data_count += 1
        self.assertTrue(data_count == len(train_ds))
        
        data_count = 0
        for _ in torch.utils.data.DataLoader(test_ds, batch_size=1, shuffle=True):
            data_count += 1
        self.assertTrue(data_count == len(test_ds))
     
if __name__ == '__main__':
    unittest.main()
    
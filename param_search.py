import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from datetime import datetime
from tqdm import tqdm
import logging
from itertools import product

# Import custom modules
from models import SRCNN_RDN
from utils import calculate_psnr, AverageMeter
import h5py
from torch.utils.data import Dataset

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('param_search.log'),
        logging.StreamHandler()
    ]
)

class EarlyStopping:
    """Early stopping to prevent overfitting"""
    def __init__(self, patience=7, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

class ParamSearcher:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.results_dir = config['results_dir']
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Initialize data loaders
        self._init_dataloaders()
        
        # Parameter search space
        self.param_grid = {
            'num_features': [32, 64, 128],
            'growth_rate': [16, 32, 48],
            'num_blocks': [3, 5, 7],
            'num_layers': [3, 5, 7]
        }
        
        self.results = []
        self._load_previous_results()

    def _init_dataloaders(self):
        """Initialize data loaders"""
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485], std=[0.229])
        ])

        class SRDataset(Dataset):
            """Custom Dataset for Super Resolution"""
            def __init__(self, h5_file, transform=None):
                self.h5_file = h5_file
                self.transform = transform
                
                with h5py.File(self.h5_file, 'r') as f:
                    # Verify the required datasets exist
                    assert 'lr' in f and 'hr' in f, "Dataset must contain 'lr' and 'hr' groups"
                    self.length = len(f['lr'])
                    # Store the shapes for verification
                    print(f"Dataset shapes - LR: {f['lr'].shape}, HR: {f['hr'].shape}")
            
            def __len__(self):
                return self.length
            
            def __getitem__(self, idx):
                if not isinstance(idx, int):
                    raise TypeError(f"Index must be an integer, got {type(idx)}")
                if idx >= self.length:
                    raise IndexError(f"Index {idx} out of bounds for dataset of length {self.length}")
                
                with h5py.File(self.h5_file, 'r') as f:
                    # Convert numpy arrays to tensors
                    lr_img = torch.from_numpy(f['lr'][idx]).float()
                    hr_img = torch.from_numpy(f['hr'][idx]).float()
                    
                    # Add channel dimension if needed (assuming grayscale images)
                    if len(lr_img.shape) == 2:
                        lr_img = lr_img.unsqueeze(0)
                    if len(hr_img.shape) == 2:
                        hr_img = hr_img.unsqueeze(0)
                
                if self.transform:
                    lr_img = self.transform(lr_img)
                    hr_img = self.transform(hr_img)
                
                return lr_img, hr_img

        # Create training and validation datasets using our custom SRDataset
        train_dataset = SRDataset(
            h5_file=os.path.join(self.config['data_dir'], self.config['train_file']),
            transform=transform
        )
        
        val_dataset = SRDataset(
            h5_file=os.path.join(self.config['data_dir'], self.config['test_file']),
            transform=transform
        )

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

    def _load_previous_results(self):
        """Load previous search results"""
        result_file = os.path.join(self.results_dir, 'search_results.json')
        if os.path.exists(result_file):
            with open(result_file, 'r') as f:
                self.results = json.load(f)
            logging.info(f"Loaded {len(self.results)} previous search results")

    def train_epoch(self, model, criterion, optimizer):
        """Train for one epoch"""
        model.train()
        total_loss = 0
        pbar = tqdm(self.train_loader, desc='Training')
        
        for lr_imgs, hr_imgs in pbar:
            lr_imgs = lr_imgs.to(self.device)
            hr_imgs = hr_imgs.to(self.device)
            
            optimizer.zero_grad()
            sr_imgs = model(lr_imgs)
            loss = criterion(sr_imgs, hr_imgs)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
        return total_loss / len(self.train_loader)

    def validate(self, model, criterion):
        """Validate the model"""
        model.eval()
        total_loss = 0
        total_psnr = 0
        
        with torch.no_grad():
            for lr_imgs, hr_imgs in tqdm(self.val_loader, desc='Validation'):
                lr_imgs = lr_imgs.to(self.device)
                hr_imgs = hr_imgs.to(self.device)
                
                sr_imgs = model(lr_imgs)
                loss = criterion(sr_imgs, hr_imgs)
                psnr = calculate_psnr(sr_imgs, hr_imgs)
                
                total_loss += loss.item()
                total_psnr += psnr.item()
        
        return (total_loss / len(self.val_loader),
                total_psnr / len(self.val_loader))

    def train_model(self, params):
        """Train a single model"""
        model = SRCNN_RDN(
            num_channels=1,
            **params
        ).to(self.device)
        
        criterion = nn.L1Loss()
        optimizer = optim.Adam(
            model.parameters(),
            lr=self.config['learning_rate']
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        early_stopping = EarlyStopping(patience=10)
        
        best_psnr = 0
        best_epoch = 0
        
        # Start training
        for epoch in range(self.config['max_epochs']):
            train_loss = self.train_epoch(model, criterion, optimizer)
            val_loss, val_psnr = self.validate(model, criterion)
            
            logging.info(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, "
                        f"Val Loss: {val_loss:.4f}, Val PSNR: {val_psnr:.2f}dB")
            
            scheduler.step(val_loss)
            early_stopping(val_loss)
            
            if val_psnr > best_psnr:
                best_psnr = val_psnr
                best_epoch = epoch
                # Save best model
                model_path = os.path.join(
                    self.results_dir,
                    f"model_f{params['num_features']}_g{params['growth_rate']}"
                    f"_b{params['num_blocks']}_l{params['num_layers']}.pth"
                )
                torch.save(model.state_dict(), model_path)
            
            if early_stopping.early_stop:
                logging.info("Early stopping triggered")
                break
        
        return best_psnr, best_epoch

    def search(self):
        """Execute parameter search"""
        param_combinations = [dict(zip(self.param_grid.keys(), v))
                            for v in product(*self.param_grid.values())]
        
        for params in param_combinations:
            # Check if this parameter combination has already been tested
            if any(r['params'] == params for r in self.results):
                logging.info(f"Skipping already tested parameters: {params}")
                continue
                
            logging.info(f"\nTesting parameters: {params}")
            best_psnr, best_epoch = self.train_model(params)
            
            result = {
                'params': params,
                'best_psnr': float(best_psnr),
                'best_epoch': best_epoch,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            self.results.append(result)
            
            # Save current results
            with open(os.path.join(self.results_dir, 'search_results.json'), 'w') as f:
                json.dump(self.results, f, indent=4)
        
        # Find best parameters
        best_result = max(self.results, key=lambda x: x['best_psnr'])
        logging.info("\nBest parameters found:")
        logging.info(json.dumps(best_result, indent=4))
        
        return best_result

def main():
    # Configuration parameters
    config = {
        'data_dir': 'dataset',  # Directory containing the h5 files
        'train_file': 'train_data.h5',  # Training data file name
        'test_file': 'test_data.h5',    # Test/validation data file name
        'results_dir': 'param_search_results',
        'batch_size': 16,
        'learning_rate': 1e-4,
        'max_epochs': 100,
    }
    
    # Create parameter searcher
    searcher = ParamSearcher(config)
    
    # Start search
    best_params = searcher.search()
    
    logging.info("Parameter search completed!")
    logging.info(f"Best parameters: {best_params['params']}")
    logging.info(f"Best PSNR: {best_params['best_psnr']:.2f}dB")

if __name__ == "__main__":
    main()
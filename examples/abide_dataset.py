#!/usr/bin/env python
"""
ABIDE Dataset Loader for BrainMamba

This script provides functions to download, preprocess, and load the ABIDE dataset
for use with the BrainMamba model. The ABIDE dataset contains resting-state fMRI data
from individuals with Autism Spectrum Disorder (ASD) and typically developing controls.

The script handles:
1. Downloading the preprocessed ABIDE data
2. Extracting time series from ROIs
3. Computing functional connectivity matrices
4. Creating PyTorch data loaders for training, validation, and testing
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import requests
import nibabel as nib
from nilearn import datasets, input_data, connectome
from tqdm import tqdm
import zipfile
import argparse
from typing import Tuple, Dict, List, Any, Optional
import io

# Add parent directory to path to import BrainMamba modules
sys.path.append('..')
from brainmamba.utils.connectivity import construct_functional_connectivity, get_functional_systems

# URLs for ABIDE data from NITRC
ABIDE_NITRC_BASE_URL = "https://fcp-indi.s3.amazonaws.com/data/Projects/ABIDE_Initiative"
ABIDE_PHENOTYPIC_URL = f"{ABIDE_NITRC_BASE_URL}/Phenotypic_V1_0b_preprocessed1.csv"

# Alternative download methods
NITRC_DATA_SOURCES = {
    "phenotypic": "https://fcp-indi.s3.amazonaws.com/data/Projects/ABIDE_Initiative/Phenotypic_V1_0b_preprocessed1.csv",
    "nitrc_ir": "https://nitrc.org/ir/data/",
    "coins": "https://coins.trendscenter.org/dataexchange/",
    "loris": "https://abide.loris.ca/",
}

class ABIDEDataset(Dataset):
    """
    PyTorch Dataset for ABIDE data.
    
    This dataset can provide either time series data, connectivity matrices, or both,
    depending on the specified mode.
    """
    
    def __init__(self, timeseries=None, connectivity=None, labels=None, 
                 functional_systems=None, transform=None, mode='both'):
        """
        Initialize the ABIDE dataset.
        
        Args:
            timeseries: Time series data of shape (n_subjects, n_rois, n_timepoints)
            connectivity: Connectivity matrices of shape (n_subjects, n_rois, n_rois)
            labels: Labels (0 for control, 1 for ASD) of shape (n_subjects,)
            functional_systems: Functional system assignments of shape (n_subjects, n_rois)
            transform: Optional transform to apply to the data
            mode: One of 'timeseries', 'connectivity', or 'both'
        """
        self.timeseries = timeseries
        self.connectivity = connectivity
        self.labels = labels
        self.functional_systems = functional_systems
        self.transform = transform
        self.mode = mode
        
        # Validate inputs
        if self.mode == 'timeseries' and self.timeseries is None:
            raise ValueError("Time series data must be provided when mode is 'timeseries'")
        if self.mode == 'connectivity' and self.connectivity is None:
            raise ValueError("Connectivity data must be provided when mode is 'connectivity'")
        if self.mode == 'both' and (self.timeseries is None or self.connectivity is None):
            raise ValueError("Both time series and connectivity data must be provided when mode is 'both'")
        if self.labels is None:
            raise ValueError("Labels must be provided")
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        sample = {}
        
        if self.mode in ['timeseries', 'both']:
            sample['timeseries'] = torch.FloatTensor(self.timeseries[idx])
        
        if self.mode in ['connectivity', 'both']:
            sample['connectivity'] = torch.FloatTensor(self.connectivity[idx])
        
        sample['label'] = torch.LongTensor([self.labels[idx]])[0]
        
        if self.functional_systems is not None:
            sample['functional_systems'] = torch.LongTensor(self.functional_systems[idx])
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample


def download_phenotypic_data(data_dir: str) -> str:
    """Download the phenotypic data file from NITRC."""
    os.makedirs(data_dir, exist_ok=True)
    pheno_file = os.path.join(data_dir, "Phenotypic_V1_0b_preprocessed1.csv")
    
    if not os.path.exists(pheno_file):
        print(f"Downloading phenotypic file from {ABIDE_PHENOTYPIC_URL}...")
        response = requests.get(ABIDE_PHENOTYPIC_URL)
        if response.status_code == 200:
            with open(pheno_file, 'wb') as f:
                f.write(response.content)
        else:
            raise Exception(f"Failed to download phenotypic data, status code: {response.status_code}")
    else:
        print("Phenotypic file already exists.")
        
    return pheno_file


def download_abide_preproc(data_dir: str,
                          pipeline: str = 'cpac',
                          strategy: str = 'filt_global',
                          atlas: str = 'cc200',
                          timeseries_only: bool = False) -> Tuple[str, pd.DataFrame]:
    """
    Download ABIDE preprocessed data from NITRC.
    
    Parameters
    ----------
    data_dir : str
        Directory where downloaded data should be stored
    pipeline : str
        Pipeline used for preprocessing the data
    strategy : str
        Noise removal strategy used for preprocessing the data
    atlas : str
        Atlas used for parcellating the data
    timeseries_only : bool
        If True, only download the timeseries data
    
    Returns
    -------
    pheno_file : str
        Path to the phenotypic data file
    subject_data : pd.DataFrame
        DataFrame containing phenotypic data
    """
    # Create data directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)
    
    # Download phenotypic data
    pheno_file = download_phenotypic_data(data_dir)
    
    # Load phenotypic data
    pheno_data = pd.read_csv(pheno_file)
    
    # Filter subjects based on preprocessing pipeline
    subject_ids = pheno_data['SUB_ID'].tolist()
    
    # Make sure we're only looking at those subjects that have functional data
    selected_indices = []
    for i, subject_id in enumerate(subject_ids):
        site = pheno_data.iloc[i]['SITE_ID']
        roi_file = os.path.join(data_dir, 'rois_' + atlas, site + '_' + str(subject_id) + '_rois_' + atlas + '.1D')
        if os.path.exists(roi_file):
            selected_indices.append(i)
    
    # Only keep subjects with data
    pheno_data = pheno_data.iloc[selected_indices]
    subject_ids = pheno_data['SUB_ID'].tolist()
    
    # Prepare download location
    download_dir = os.path.join(data_dir, 'rois_' + atlas)
    os.makedirs(download_dir, exist_ok=True)
    
    # Inform user about download options
    print("ABIDE data will be downloaded. Due to recent changes in data access, please use one of these options:")
    print("1. NITRC-IR: Visit https://nitrc.org/ir/app/template/Index.vm and select ABIDE")
    print("2. COINS: Visit https://coins.trendscenter.org/dataexchange/")
    print("3. LORIS: Visit https://abide.loris.ca/ (login: abide, password: abide_2012)")
    print("\nFor automatic downloads, we'll attempt to use NITRC's S3 bucket first.")
    
    # Try automatic download
    print(f"Downloading rois_{atlas} data...")
    
    missing_files = []
    for i, subject_id in enumerate(tqdm(subject_ids)):
        site = pheno_data.iloc[i]['SITE_ID']
        roi_file = os.path.join(download_dir, site + '_' + str(subject_id) + '_rois_' + atlas + '.1D')
        
        if not os.path.exists(roi_file):
            roi_url = f"{ABIDE_NITRC_BASE_URL}/{pipeline}/{strategy}/rois_{atlas}/{site}_{subject_id}_rois_{atlas}.1D"
            try:
                r = requests.get(roi_url)
                if r.status_code == 200:
                    with open(roi_file, 'wb') as f:
                        f.write(r.content)
                else:
                    missing_files.append(f"{site}_{subject_id}")
                    print(f"Failed to download {roi_url}, status code: {r.status_code}")
            except Exception as e:
                missing_files.append(f"{site}_{subject_id}")
                print(f"Error downloading {roi_url}: {e}")
    
    if missing_files:
        print("\nSome files could not be downloaded automatically.")
        print("Please download the missing files manually from one of these sources:")
        print("1. NITRC-IR: https://nitrc.org/ir/app/template/Index.vm")
        print("2. COINS: https://coins.trendscenter.org/dataexchange/")
        print("3. LORIS: https://abide.loris.ca/")
        print("\nMissing files:")
        for file in missing_files[:10]:
            print(f"- {file}_rois_{atlas}.1D")
        if len(missing_files) > 10:
            print(f"...and {len(missing_files) - 10} more")
    
    return pheno_file, pheno_data


def load_abide_data(data_dir: str,
                   pipeline: str = 'cpac',
                   strategy: str = 'filt_global',
                   atlas: str = 'cc200') -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[int], pd.DataFrame]:
    """
    Load ABIDE data from specified directory.
    Will download the data if it doesn't exist.
    
    Parameters
    ----------
    data_dir : str
        Directory where the data is stored
    pipeline : str
        Pipeline used for preprocessing the data
    strategy : str
        Noise removal strategy used for preprocessing the data
    atlas : str
        Atlas used for parcellating the data
    
    Returns
    -------
    timeseries : np.ndarray
        Timeseries data for all subjects
    connectivity : np.ndarray
        Connectivity matrices for all subjects
    labels : np.ndarray
        Labels for all subjects (0 for control, 1 for autism)
    subject_ids : List[int]
        List of subject IDs
    phenotypic : pd.DataFrame
        Phenotypic data for all subjects
    """
    # Check if data exists, download if it doesn't
    pheno_file = os.path.join(data_dir, 'Phenotypic_V1_0b_preprocessed1.csv')
    if not os.path.exists(pheno_file):
        print("Downloading ABIDE data...")
        pheno_file, phenotypic = download_abide_preproc(
            data_dir=data_dir,
            pipeline=pipeline,
            strategy=strategy,
            atlas=atlas
        )
    else:
        print("Loading existing ABIDE data...")
        phenotypic = pd.read_csv(pheno_file)
    
    # Find all available subjects and load their data
    subject_ids = []
    timeseries_data = []
    labels = []
    
    for i, row in phenotypic.iterrows():
        subject_id = row['SUB_ID']
        site = row['SITE_ID']
        roi_file = os.path.join(data_dir, 'rois_' + atlas, site + '_' + str(subject_id) + '_rois_' + atlas + '.1D')
        
        if os.path.exists(roi_file):
            try:
                ts = np.loadtxt(roi_file)
                if ts.shape[0] > 0:  # Make sure we have data
                    subject_ids.append(subject_id)
                    timeseries_data.append(ts)
                    labels.append(1 if row['DX_GROUP'] == 1 else 0)  # 1 for autism, 0 for control
            except Exception as e:
                print(f"Error loading {roi_file}: {e}")
    
    # Convert to numpy arrays
    timeseries = np.array(timeseries_data)
    labels = np.array(labels)
    
    # Compute connectivity matrices
    connectivity = np.zeros((len(timeseries), timeseries[0].shape[1], timeseries[0].shape[1]))
    for i, ts in enumerate(timeseries):
        corr = np.corrcoef(ts.T)
        # Set diagonal to 0
        np.fill_diagonal(corr, 0)
        connectivity[i] = corr
    
    print(f"Loaded {len(subject_ids)} subjects.")
    print(f"Timeseries shape: {timeseries.shape}")
    print(f"Connectivity shape: {connectivity.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Label distribution: {np.bincount(labels)}")
    
    return timeseries, connectivity, labels, subject_ids, phenotypic


def create_abide_dataloaders(timeseries: np.ndarray,
                           connectivity: np.ndarray,
                           labels: np.ndarray,
                           batch_size: int = 32,
                           train_ratio: float = 0.7,
                           val_ratio: float = 0.15,
                           mode: str = 'both',
                           seed: int = 42) -> Tuple[DataLoader, DataLoader, DataLoader, int, int, int]:
    """
    Create train, validation, and test dataloaders for ABIDE data.
    
    Parameters
    ----------
    timeseries : np.ndarray
        Timeseries data for all subjects
    connectivity : np.ndarray
        Connectivity matrices for all subjects
    labels : np.ndarray
        Labels for all subjects
    batch_size : int
        Batch size for dataloaders
    train_ratio : float
        Ratio of data to use for training
    val_ratio : float
        Ratio of data to use for validation
    mode : str
        Mode for the dataset, can be 'timeseries', 'connectivity', or 'both'
    seed : int
        Random seed for reproducibility
    
    Returns
    -------
    train_loader : DataLoader
        DataLoader for training data
    val_loader : DataLoader
        DataLoader for validation data
    test_loader : DataLoader
        DataLoader for test data
    num_classes : int
        Number of classes
    num_nodes : int
        Number of nodes in the brain network
    seq_len : int
        Length of the timeseries
    """
    dataset = ABIDEDataset(timeseries, connectivity, labels, mode=mode)
    
    # Split dataset
    torch.manual_seed(seed)
    n_samples = len(dataset)
    n_train = int(n_samples * train_ratio)
    n_val = int(n_samples * val_ratio)
    n_test = n_samples - n_train - n_val
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [n_train, n_val, n_test]
    )
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Get dataset properties
    num_classes = len(torch.unique(labels))
    num_nodes = connectivity.shape[1]
    seq_len = timeseries.shape[2] if len(timeseries.shape) > 2 else 0
    
    return train_loader, val_loader, test_loader, num_classes, num_nodes, seq_len


def main():
    """
    Main function to download and process ABIDE data.
    """
    parser = argparse.ArgumentParser(description='Download and preprocess ABIDE data')
    parser.add_argument('--data_dir', type=str, default='data/abide', 
                        help='Directory to store the ABIDE data')
    parser.add_argument('--pipeline', type=str, default='cpac',
                        help='Pipeline used for preprocessing')
    parser.add_argument('--strategy', type=str, default='filt_global',
                        help='Noise removal strategy used for preprocessing')
    parser.add_argument('--atlas', type=str, default='cc200',
                        help='Atlas used for parcellating')
    args = parser.parse_args()
    
    # Download and load ABIDE data
    pheno_file = download_abide_preproc(
        data_dir=args.data_dir,
        pipeline=args.pipeline,
        strategy=args.strategy,
        atlas=args.atlas
    )


if __name__ == '__main__':
    main() 
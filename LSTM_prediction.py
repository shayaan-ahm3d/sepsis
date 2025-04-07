import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, precision_recall_curve, average_precision_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Set up logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set seeds for reproducibility
def set_seeds(seed=42):
    """Set seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

#################################################
# Data Loading and Preprocessing
#################################################

def load_patient_data(base_directory=None):
    """
    Load all patient PSV files from training_setA and training_setB directories
    and track which dataset each patient comes from
    
    Args:
        base_directory (str, optional): Base directory containing training_setA and training_setB
        If None, will use the directory of this script.
        
    Returns:
        tuple: (list of dataframes, list of patient IDs, list of dataset sources)
    """
    patients = []
    patient_ids = []
    dataset_sources = []  # New list to track which dataset each patient is from
    
    # If no base directory is provided, use the directory of this script
    if base_directory is None:
        base_directory = os.path.dirname(os.path.abspath(__file__))
    
    logger.info(f"Base directory: {base_directory}")
    
    # Define the directories to search
    directories = [
        os.path.join(base_directory, "training_setA"),
        os.path.join(base_directory, "training_setB")
    ]
    
    for directory in directories:
        dataset_name = os.path.basename(directory)  # Get 'training_setA' or 'training_setB'
        logger.info(f"Looking for PSV files in {directory}")
        
        # Check if directory exists
        if not os.path.exists(directory):
            logger.warning(f"Directory {directory} not found. Skipping.")
            continue
        
        # Get list of PSV files
        psv_files = [f for f in os.listdir(directory) if f.endswith('.psv')]
        
        if not psv_files:
            logger.warning(f"No PSV files found in {directory}. Skipping.")
            continue
        
        logger.info(f"Found {len(psv_files)} PSV files in {directory}")
        
        for filename in psv_files:
            patient_id = filename.split('.')[0]  # Extract patient ID from filename
            file_path = os.path.join(directory, filename)
            
            try:
                # Load PSV file with pipe delimiter
                df = pd.read_csv(file_path, sep='|')
                
                # Add patient ID column
                df['patient_id'] = patient_id
                
                patients.append(df)
                patient_ids.append(patient_id)
                dataset_sources.append(dataset_name)  # Track which dataset this patient is from
            except Exception as e:
                logger.warning(f"Error loading {filename}: {e}")
    
    logger.info(f"Successfully loaded {len(patients)} patient records")
    
    if len(patients) == 0:
        raise ValueError("No PSV files were loaded. Please check your directory structure.")
    
    return patients, patient_ids, dataset_sources

def sample_balanced_dataset(patients, patient_ids, dataset_sources, subset_size=5000):
    """
    Sample a balanced subset of patients from both training sets
    
    Args:
        patients (list): List of patient dataframes
        patient_ids (list): List of patient IDs
        dataset_sources (list): List of dataset sources for each patient
        subset_size (int): Total desired subset size (will be balanced between datasets)
    
    Returns:
        tuple: (subset of patients, subset of patient IDs)
    """
    # Create indices for each dataset
    setA_indices = [i for i, source in enumerate(dataset_sources) if 'setA' in source]
    setB_indices = [i for i, source in enumerate(dataset_sources) if 'setB' in source]
    
    logger.info(f"Original distribution: {len(setA_indices)} patients from set A, "
                f"{len(setB_indices)} patients from set B")
    
    # Determine sample size per dataset (balanced)
    per_dataset_size = subset_size // 2
    
    # Adjust if one dataset has fewer samples than requested
    if len(setA_indices) < per_dataset_size:
        logger.warning(f"Training set A has fewer samples ({len(setA_indices)}) than requested "
                       f"({per_dataset_size}). Using all available samples.")
        # Take all from set A and more from set B
        sampled_setA_indices = setA_indices
        remaining = subset_size - len(setA_indices)
        sampled_setB_indices = np.random.choice(setB_indices, min(remaining, len(setB_indices)), replace=False)
    elif len(setB_indices) < per_dataset_size:
        logger.warning(f"Training set B has fewer samples ({len(setB_indices)}) than requested "
                       f"({per_dataset_size}). Using all available samples.")
        # Take all from set B and more from set A
        sampled_setB_indices = setB_indices
        remaining = subset_size - len(setB_indices)
        sampled_setA_indices = np.random.choice(setA_indices, min(remaining, len(setA_indices)), replace=False)
    else:
        # Standard case: equal sampling from both datasets
        sampled_setA_indices = np.random.choice(setA_indices, per_dataset_size, replace=False)
        sampled_setB_indices = np.random.choice(setB_indices, per_dataset_size, replace=False)
    
    # Combine sampled indices
    sampled_indices = np.concatenate([sampled_setA_indices, sampled_setB_indices])
    
    # Create subset
    subset_patients = [patients[i] for i in sampled_indices]
    subset_patient_ids = [patient_ids[i] for i in sampled_indices]
    
    # Count patients from each dataset in the subset
    subset_setA_count = sum(1 for i in sampled_indices if 'setA' in dataset_sources[i])
    subset_setB_count = sum(1 for i in sampled_indices if 'setB' in dataset_sources[i])
    
    logger.info(f"Created balanced subset with {subset_setA_count} patients from set A, "
                f"{subset_setB_count} patients from set B")
    
    return subset_patients, subset_patient_ids


def add_time_features(patients):
    """
    Add time-based features to help the model understand temporal patterns
    
    Args:
        patients (list): List of patient dataframes
        
    Returns:
        list: List of patient dataframes with time features
    """
    logger.info("Adding time-based features")
    
    for i, df in enumerate(patients):
        # Time since hospital admission
        if 'HospAdmTime' in df.columns and 'ICULOS' in df.columns:
            patients[i]['time_since_admission'] = df['ICULOS'] - df['HospAdmTime']
        
        # Hour of day might capture circadian rhythms
        if 'ICULOS' in df.columns:
            patients[i]['hour_sin'] = np.sin(2 * np.pi * (df['ICULOS'] % 24) / 24)
            patients[i]['hour_cos'] = np.cos(2 * np.pi * (df['ICULOS'] % 24) / 24)
        
    return patients


def add_clinical_features(patients):
    """
    Add clinically relevant sepsis-specific features
    
    Args:
        patients (list): List of patient dataframes
        
    Returns:
        list: Patient dataframes with clinical features
    """
    logger.info("Adding sepsis-specific clinical features")
    
    for i, df in enumerate(patients):
        # Create a copy to ensure we're working with a fresh DataFrame
        patients[i] = df.copy()
        
        # Vital sign trends and volatility
        for vital in ['HR', 'SBP', 'MAP', 'DBP', 'Temp', 'Resp', 'O2Sat']:
            if vital in df.columns:
                # Rolling statistics with different windows
                if len(df) >= 3:
                    # Shorter trend for rapid changes
                    patients[i][f'{vital}_trend_3h'] = df[vital].rolling(window=3, min_periods=1).mean().fillna(df[vital])
                    # Calculate rate of change
                    patients[i][f'{vital}_slope'] = df[vital].diff().fillna(0)
                
                if len(df) >= 6:
                    # Volatility measure
                    patients[i][f'{vital}_volatility'] = df[vital].rolling(window=6, min_periods=1).std().fillna(0)
        
        # Shock Index: Heart Rate / Systolic BP (clinical predictor of sepsis)
        if 'HR' in df.columns and 'SBP' in df.columns:
            with np.errstate(divide='ignore', invalid='ignore'):
                shock_index = df['HR'] / df['SBP']
            patients[i]['ShockIndex'] = np.where(np.isfinite(shock_index), shock_index, np.nan)
        
        # SOFA score components (simplified)
        # Cardiovascular
        if 'MAP' in df.columns:
            patients[i]['SOFA_cardio'] = np.where(df['MAP'] < 70, 1, 0)
        
        # Coagulation
        if 'Platelets' in df.columns:
            patients[i]['SOFA_coag'] = np.where(df['Platelets'] < 150, 1, 
                                           np.where(df['Platelets'] < 100, 2, 0))
        
        # Liver
        if 'Bilirubin_total' in df.columns:
            patients[i]['SOFA_liver'] = np.where(df['Bilirubin_total'] > 1.2, 1,
                                            np.where(df['Bilirubin_total'] > 2, 2, 0))
        
        # Renal
        if 'Creatinine' in df.columns:
            patients[i]['SOFA_renal'] = np.where(df['Creatinine'] > 1.2, 1,
                                            np.where(df['Creatinine'] > 2, 2, 0))
        
        # Lab value ratios (clinically relevant)
        if 'BUN' in df.columns and 'Creatinine' in df.columns:
            with np.errstate(divide='ignore', invalid='ignore'):
                bun_cr_ratio = df['BUN'] / df['Creatinine']
            patients[i]['BUN_Creatinine_ratio'] = np.where(np.isfinite(bun_cr_ratio), bun_cr_ratio, np.nan)
        
        # WBC trends are important for sepsis
        if 'WBC' in df.columns:
            patients[i]['WBC_is_low'] = np.where(df['WBC'] < 4, 1, 0)
            patients[i]['WBC_is_high'] = np.where(df['WBC'] > 12, 1, 0)
            
        # Lactate is a key sepsis biomarker
        if 'Lactate' in df.columns:
            patients[i]['Lactate_high'] = np.where(df['Lactate'] > 2, 1, 0)
        
        # Add time since last measurement for lab values
        for lab in ['WBC', 'Lactate', 'Creatinine', 'Platelets', 'Bilirubin_total']:
            if lab in df.columns:
                # Create mask for when values change (indicating new measurement)
                value_change = df[lab].diff() != 0
                # Time since last measurement
                time_col = f'{lab}_time_since_measured'
                # Initialize the column with zeros
                patients[i][time_col] = 0
                
                # Calculate time since last measurement
                last_measured = 0
                # Here's the key change - pre-create the array, then assign it at once
                time_values = np.zeros(len(df))
                
                for idx in range(len(df)):
                    if not pd.isna(df[lab].iloc[idx]) and (idx == 0 or value_change.iloc[idx]):
                        last_measured = 0
                    else:
                        last_measured += 1
                    # Store in our temporary array instead of assigning directly
                    time_values[idx] = last_measured
                
                # Assign the entire array at once
                patients[i][time_col] = time_values
    
    return patients



def preprocess_patient_data(patients, max_seq_length=None):
    """
    Preprocess patient data with consistent features across all patients
    """
    logger.info("Preprocessing patient data")
    
    # Define categorical columns based on documentation
    categorical_cols = ['Gender', 'Unit1', 'Unit2']
    
    # Identify all possible numerical columns across all patients
    all_numerical_cols = set()
    for df in patients:
        numerical_in_df = df.select_dtypes(include=['float64', 'int64']).columns
        all_numerical_cols.update(numerical_in_df)
    
    # Remove non-feature columns
    for col in ['patient_id', 'SepsisLabel']:
        if col in all_numerical_cols:
            all_numerical_cols.remove(col)
    
    all_numerical_cols = list(all_numerical_cols)
    
    # Identify all missing indicators we'll need to create
    missing_indicators = set()
    for df in patients:
        for col in all_numerical_cols:
            if col in df.columns and df[col].isnull().any():
                missing_indicators.add(f"{col}_missing")
    
    # All features including missing indicators
    all_feature_cols = all_numerical_cols + list(missing_indicators)
    
    logger.info(f"Identified {len(all_numerical_cols)} numerical features, {len(categorical_cols)} categorical features, and {len(missing_indicators)} missing indicators")
    
    # Create consistent dataframes for all patients
    all_numerical_data = []
    all_categorical_data = []
    
    for df in patients:
        # Create numerical features all at once instead of column by column
        # First, create a dictionary to hold all column data
        num_data = {}
        
        # Add numerical columns
        for col in all_numerical_cols:
            if col in df.columns:
                num_data[col] = df[col]
            else:
                num_data[col] = np.nan
        
        # Add missing indicators all at once
        for indicator in missing_indicators:
            base_col = indicator.replace("_missing", "")
            if base_col in df.columns:
                num_data[indicator] = df[base_col].isnull().astype(float)
            else:
                num_data[indicator] = 0.0  # No missing values for this column
        
        # Create the dataframe all at once
        num_df = pd.DataFrame(num_data, index=df.index)
        
        # Fill remaining missing values
        num_df = num_df.fillna(0)
        all_numerical_data.append(num_df)
        
        # Process categorical features similarly
        cat_data = {}
        for col in categorical_cols:
            if col in df.columns:
                # Convert to string for consistency
                cat_data[col] = df[col].astype(str)
            else:
                cat_data[col] = 'missing'
        
        cat_df = pd.DataFrame(cat_data, index=df.index)
        all_categorical_data.append(cat_df)
    
    # The rest of the function remains unchanged
    # Fit preprocessors on combined data
    logger.info("Fitting preprocessors on all data")
    
    # Concatenate to fit scalers/encoders
    all_num_df = pd.concat(all_numerical_data)
    all_cat_df = pd.concat(all_categorical_data)
    
    # Fit numerical scaler
    numerical_scaler = StandardScaler()
    numerical_scaler.fit(all_num_df)
    
    # Fit categorical encoder
    try:
        categorical_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    except TypeError:
        categorical_encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
    
    categorical_encoder.fit(all_cat_df)
    
    # Transform each patient's data
    logger.info("Transforming each patient's data")
    
    processed_patients = []
    sepsis_labels = []
    patient_ids = []
    
    for i, df in enumerate(patients):
        # Store patient ID
        patient_ids.append(df['patient_id'].iloc[0] if 'patient_id' in df.columns else f"unknown_{i}")
        
        # Transform features
        num_scaled = numerical_scaler.transform(all_numerical_data[i])
        cat_encoded = categorical_encoder.transform(all_categorical_data[i])
        
        # Combine features
        combined_features = np.hstack([num_scaled, cat_encoded])
        
        # Get sepsis labels
        sepsis = df['SepsisLabel'].values if 'SepsisLabel' in df.columns else np.zeros(len(df))
        
        # Pad or truncate sequences
        if max_seq_length:
            if len(combined_features) > max_seq_length:
                combined_features = combined_features[-max_seq_length:]
                sepsis = sepsis[-max_seq_length:]
            elif len(combined_features) < max_seq_length:
                padding = np.zeros((max_seq_length - len(combined_features), combined_features.shape[1]))
                combined_features = np.vstack([padding, combined_features])
                
                sepsis_padding = np.zeros(max_seq_length - len(sepsis))
                sepsis = np.concatenate([sepsis_padding, sepsis])
        
        processed_patients.append(combined_features)
        sepsis_labels.append(sepsis)
    
    # Create feature names for interpretability
    feature_names = list(all_num_df.columns)
    for cat_feature, categories in zip(categorical_cols, categorical_encoder.categories_):
        for category in categories:
            feature_names.append(f"{cat_feature}_{category}")
    
    logger.info(f"Processed {len(processed_patients)} patients with {len(feature_names)} features")
    
    return processed_patients, sepsis_labels, patient_ids, feature_names




#################################################
# PyTorch Dataset and DataLoader
#################################################

class SepsisDataset(Dataset):
    """PyTorch Dataset for sepsis prediction"""
    def __init__(self, sequences, labels):
        self.sequences = [torch.FloatTensor(seq) for seq in sequences]
        self.labels = [torch.FloatTensor(label) for label in labels]
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

def collate_fn(batch):
    """
    Custom collate function for variable length sequences
    
    Args:
        batch: List of (sequence, label) tuples
        
    Returns:
        tuple: (sequences_tensor, labels_tensor, lengths)
    """
    # Sort batch by sequence length (descending)
    batch.sort(key=lambda x: len(x[0]), reverse=True)
    
    # Separate sequences and labels
    sequences, labels = zip(*batch)
    
    # Get sequence lengths
    lengths = [len(seq) for seq in sequences]
    
    # Convert to tensors
    sequences_tensor = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True)
    labels_tensor = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True)
    
    return sequences_tensor, labels_tensor, lengths

#################################################
# Model Definition
#################################################

class BidirectionalLSTMTimeStep(nn.Module):
    """PyTorch implementation of Bidirectional LSTM for time step-level sepsis prediction"""
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.4):
        super(BidirectionalLSTMTimeStep, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Bidirectional LSTM layers
        self.lstm = nn.LSTM(
            input_dim, 
            hidden_dim, 
            num_layers=num_layers, 
            batch_first=True, 
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Self-attention mechanism
        self.attention_query = nn.Linear(hidden_dim * 2, hidden_dim * 2)
        self.attention_key = nn.Linear(hidden_dim * 2, hidden_dim * 2)
        self.attention_value = nn.Linear(hidden_dim * 2, hidden_dim * 2)
        
        # Final layers - applied to each time step individually
        self.fc1 = nn.Linear(hidden_dim * 2, 32)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(32, 1)
        
    def attention(self, query, key, value):
        """Custom attention mechanism"""
        # Calculate attention scores
        energy = torch.bmm(query, key.transpose(1, 2))
        
        # Scale dot-product attention
        scaling_factor = torch.sqrt(torch.tensor(key.size(-1), dtype=torch.float32, device=key.device))
        energy = energy / scaling_factor
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(energy, dim=-1)
        
        # Apply attention weights to values
        out = torch.bmm(attention_weights, value)
        
        return out, attention_weights
        
    def forward(self, x, lengths=None, return_attention=False):
        batch_size = x.size(0)
        
        # Apply LSTM
        if lengths is not None:
            # Pack padded sequence for more efficient computation
            packed_input = nn.utils.rnn.pack_padded_sequence(
                x, lengths, batch_first=True, enforce_sorted=True
            )
            packed_output, _ = self.lstm(packed_input)
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        else:
            lstm_out, _ = self.lstm(x)
        
        # Apply attention
        query = self.attention_query(lstm_out)
        key = self.attention_key(lstm_out)
        value = self.attention_value(lstm_out)
        
        context, attention_weights = self.attention(query, key, value)
        
        # Apply dense layers to each time step independently
        # No pooling - we keep the time dimension intact
        dense1 = F.relu(self.fc1(context))
        dense1 = self.dropout(dense1)
        logits = self.fc2(dense1)
        
        # Apply sigmoid for binary classification at each time step
        outputs = torch.sigmoid(logits)
        
        if return_attention:
            return outputs, attention_weights
        else:
            return outputs

#################################################
# Training Function
#################################################

def sepsis_weighted_bce_loss(predictions, targets, lengths, class_weight=32.0, time_discount=0.9):
    """
    Custom loss function for sepsis prediction that:
    1. Handles class imbalance
    2. Puts more weight on predictions closer to sepsis onset
    3. Only considers valid time steps based on sequence lengths
    
    Args:
        predictions: Model predictions [batch_size, seq_len, 1]
        targets: Ground truth labels [batch_size, seq_len]
        lengths: Actual sequence lengths
        class_weight: Weight for positive class (sepsis)
        time_discount: How much to discount predictions farther from sepsis onset
        
    Returns:
        Loss value
    """
    device = predictions.device
    batch_size = predictions.size(0)
    
    # Initialize total loss
    total_loss = 0
    total_elements = 0
    
    # Process each sequence separately
    for i in range(batch_size):
        # Extract valid sequence for this patient
        seq_len = lengths[i]
        seq_preds = predictions[i, :seq_len, 0]
        seq_targets = targets[i, :seq_len]
        
        # Check if the patient develops sepsis
        sepsis_indices = (seq_targets > 0).nonzero(as_tuple=True)[0]
        
        if len(sepsis_indices) > 0:
            # This patient develops sepsis at some point
            first_sepsis_idx = sepsis_indices[0].item()
            
            # Create weights tensor
            weights = torch.ones_like(seq_targets)
            
            # Apply weights
            for t in range(seq_len):
                if seq_targets[t] > 0:
                    # Positive examples (sepsis) get the class weight
                    weights[t] = class_weight
                else:
                    # Negative examples get weighted based on proximity to sepsis
                    if t < first_sepsis_idx:
                        # Before sepsis - weight increases as we approach onset
                        distance_to_sepsis = first_sepsis_idx - t
                        # Discount based on distance (higher weight closer to sepsis)
                        weights[t] = 1.0 * (time_discount ** distance_to_sepsis)
                    else:
                        # After sepsis - these are likely data errors, give low weight
                        weights[t] = 0.5
        else:
            # Patient never develops sepsis, use standard weight
            weights = torch.ones_like(seq_targets)
            
        # Calculate BCE loss with instance weights
        bce_loss = F.binary_cross_entropy(seq_preds, seq_targets, reduction='none')
        weighted_loss = bce_loss * weights
        
        # Add to total
        total_loss += weighted_loss.sum()
        total_elements += seq_len
    
    # Return average loss
    return total_loss / max(total_elements, 1)




def train_model(train_loader, val_loader, input_dim, device, 
                class_weight=32.0, epochs=30, lr=0.001):
    """
    Train the sepsis prediction model with time step-level predictions
    
    Args:
        train_loader: Training data loader
        val_loader: Validation data loader
        input_dim: Input feature dimension
        device: PyTorch device
        class_weight: Weight for positive class
        epochs: Number of training epochs
        lr: Learning rate
        
    Returns:
        tuple: (model, history)
    """
    logger.info(f"Training time step-level sepsis prediction model on device: {device}")
    
    # Create model and move to device
    model = BidirectionalLSTMTimeStep(input_dim=input_dim)
    model.to(device)
    
    # Define optimizer with weight decay
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    
    # Define learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, min_lr=0.00001
    )
    
    # For early stopping
    best_val_auc = 0
    patience_counter = 0
    best_model_state = None
    early_stopping_patience = 10
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_auc': [],
        'val_auc': [],
        'train_auprc': [],
        'val_auprc': []
    }
    
    logger.info(f"Starting training for {epochs} epochs")
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_losses = []
        all_train_preds = []
        all_train_targets = []
        
        for inputs, targets, lengths in train_loader:
            # Move to device
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(inputs, lengths)  # Shape: [batch_size, seq_len, 1]
            
            # Calculate loss with clinical relevance
            loss = sepsis_weighted_bce_loss(outputs, targets, lengths, class_weight)
            
            # Backward pass and optimize
            loss.backward()
            
            # Gradient clipping to prevent explosion
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Store for metrics calculation
            train_losses.append(loss.item())
            
            # Collect predictions and targets for metrics
            for i, length in enumerate(lengths):
                all_train_preds.extend(outputs[i, :length, 0].detach().cpu().numpy())
                all_train_targets.extend(targets[i, :length].cpu().numpy())
        
        # Validation phase
        model.eval()
        val_losses = []
        all_val_preds = []
        all_val_targets = []
        
        with torch.no_grad():
            for inputs, targets, lengths in val_loader:
                # Move to device
                inputs = inputs.to(device)
                targets = targets.to(device)
                
                # Forward pass
                outputs = model(inputs, lengths)
                
                # Calculate loss
                loss = sepsis_weighted_bce_loss(outputs, targets, lengths, class_weight)
                
                # Store for metrics calculation
                val_losses.append(loss.item())
                
                # Collect predictions and targets for metrics
                for i, length in enumerate(lengths):
                    all_val_preds.extend(outputs[i, :length, 0].detach().cpu().numpy())
                    all_val_targets.extend(targets[i, :length].cpu().numpy())
        
        # Calculate epoch metrics
        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        
        # Calculate AUC and AUPRC if possible
        train_auc = roc_auc_score(all_train_targets, all_train_preds) if len(set(all_train_targets)) > 1 else 0.5
        val_auc = roc_auc_score(all_val_targets, all_val_preds) if len(set(all_val_targets)) > 1 else 0.5
        
        # AUPRC (more relevant for imbalanced datasets)
        train_auprc = average_precision_score(all_train_targets, all_train_preds) if len(set(all_train_targets)) > 1 else 0
        val_auprc = average_precision_score(all_val_targets, all_val_preds) if len(set(all_val_targets)) > 1 else 0
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_auc'].append(train_auc)
        history['val_auc'].append(val_auc)
        history['train_auprc'].append(train_auprc)
        history['val_auprc'].append(val_auprc)
        
        # Update learning rate scheduler based on validation AUC
        scheduler.step(val_auc)
        
        # Early stopping based on validation AUC
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            logger.info(f"Epoch {epoch+1}: New best validation AUC: {val_auc:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        # Print progress
        logger.info(f"Epoch {epoch+1}/{epochs} - "
              f"Train Loss: {train_loss:.4f}, Train AUC: {train_auc:.4f}, Train AUPRC: {train_auprc:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}, Val AUPRC: {val_auprc:.4f}")
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        logger.info("Loaded best model from checkpoint")
    
    return model, history

##############################
# Evaluation Functions
##############################

def evaluate_model(model, test_loader, device, patient_ids_test, threshold=0.5):
    """
    Evaluate model with time step-level predictions and clinical metrics
    
    Args:
        model: Trained PyTorch model
        test_loader: Test data loader
        device: PyTorch device
        patient_ids_test: List of patient IDs for test set
        threshold: Classification threshold
        
    Returns:
        dict: Evaluation results
    """
    logger.info("Evaluating time step-level sepsis prediction model")
    model.eval()
    
    # Initialize lists to store predictions and targets
    all_preds = []
    all_targets = []
    patient_predictions = {}
    
    # Initialize patient tracking
    current_patient_idx = 0
    
    with torch.no_grad():
        for inputs, targets, lengths in test_loader:
            # Move to device
            inputs = inputs.to(device)
            
            # Forward pass
            outputs = model(inputs, lengths)  # Shape: [batch_size, seq_len, 1]
            
            # Convert to numpy for further analysis
            outputs_np = outputs.cpu().numpy()
            targets_np = targets.cpu().numpy()
            
            # Store predictions for each patient
            for i in range(inputs.size(0)):
                patient_id = patient_ids_test[current_patient_idx]
                current_patient_idx += 1
                
                # Get actual sequence length
                seq_len = lengths[i]
                
                # Store the predictions and actual values
                patient_predictions[patient_id] = {
                    'predictions': outputs_np[i, :seq_len, 0],  # Time step-level predictions
                    'targets': targets_np[i, :seq_len]
                }
                
                # Add to overall lists
                all_preds.extend(outputs_np[i, :seq_len, 0])
                all_targets.extend(targets_np[i, :seq_len])
    
    # Initialize metrics
    results = {
        'patient_id': [],
        'actual_sepsis': [],
        'predicted_sepsis': [],
        'time_to_sepsis': [],
        'prediction_lead_time': [],
        'false_alarm_rate': [],
        'detection_delay': []
    }
    
    # Evaluate each patient separately
    for patient_id, data in patient_predictions.items():
        preds = data['predictions']
        targets = data['targets']
        
        # Binary predictions based on threshold
        binary_preds = (preds > threshold).astype(int)
        
        # Check if patient actually developed sepsis
        actual_sepsis = int(np.max(targets) > 0)
        
        # Check if model predicted sepsis
        predicted_sepsis = int(np.max(binary_preds) > 0)
        
        # Find time to sepsis
        time_to_sepsis = -1
        if actual_sepsis:
            sepsis_indices = np.where(targets > 0)[0]
            if len(sepsis_indices) > 0:
                time_to_sepsis = sepsis_indices[0]
        
        # Calculate clinical metrics
        prediction_lead_time = -1
        detection_delay = -1
        false_alarm_rate = 0
        
        if predicted_sepsis:
            # Find first prediction above threshold
            alert_indices = np.where(binary_preds > 0)[0]
            
            if len(alert_indices) > 0:
                first_alert = alert_indices[0]
                
                if actual_sepsis:
                    # Calculate lead time or delay
                    if first_alert < time_to_sepsis:
                        # Early prediction
                        prediction_lead_time = time_to_sepsis - first_alert
                    else:
                        # Late prediction
                        detection_delay = first_alert - time_to_sepsis
                else:
                    # False positive
                    false_alarm_rate = len(alert_indices) / len(targets)
        
        # Store results
        results['patient_id'].append(patient_id)
        results['actual_sepsis'].append(actual_sepsis)
        results['predicted_sepsis'].append(predicted_sepsis)
        results['time_to_sepsis'].append(time_to_sepsis)
        results['prediction_lead_time'].append(prediction_lead_time)
        results['false_alarm_rate'].append(false_alarm_rate)
        results['detection_delay'].append(detection_delay)
    
    # Convert to binary for classification metrics
    all_preds_binary = np.array(all_preds) > threshold
    
    # Calculate metrics
    try:
        report = classification_report(all_targets, all_preds_binary)
        auc = roc_auc_score(all_targets, all_preds)
        cm = confusion_matrix(all_targets, all_preds_binary)
        
        # Calculate AUPRC
        precision, recall, _ = precision_recall_curve(all_targets, all_preds)
        auprc = average_precision_score(all_targets, all_preds)
    except Exception as e:
        logger.warning(f"Error calculating metrics: {e}")
        report = "Could not calculate metrics"
        auc = 0
        cm = np.zeros((2, 2))
        precision, recall = np.array([]), np.array([])
        auprc = 0
    
    # Calculate early detection metrics
    df_results = pd.DataFrame(results)
    sepsis_cases = df_results[df_results['actual_sepsis'] == 1]
    detected_cases = sepsis_cases[sepsis_cases['predicted_sepsis'] == 1]
    
    early_detections = detected_cases[detected_cases['prediction_lead_time'] > 0]
    
    # Calculate detection rate
    detection_rate = len(detected_cases) / len(sepsis_cases) if len(sepsis_cases) > 0 else 0
    early_detection_rate = len(early_detections) / len(sepsis_cases) if len(sepsis_cases) > 0 else 0
    
    # Calculate average lead time
    avg_lead_time = early_detections['prediction_lead_time'].mean() if len(early_detections) > 0 else 0
    
    # Calculate false alarm rate
    non_sepsis_cases = df_results[df_results['actual_sepsis'] == 0]
    false_alarms = non_sepsis_cases[non_sepsis_cases['predicted_sepsis'] == 1]
    false_alarm_rate = len(false_alarms) / len(non_sepsis_cases) if len(non_sepsis_cases) > 0 else 0
    
    # Log results
    logger.info(f"Evaluation complete. AUC: {auc:.4f}, AUPRC: {auprc:.4f}")
    logger.info(f"Detection rate: {detection_rate:.2f}, Early detection rate: {early_detection_rate:.2f}")
    logger.info(f"Average lead time: {avg_lead_time:.2f} hours for early detections")
    logger.info(f"False alarm rate: {false_alarm_rate:.2f}")
    
    return {
        'patient_results': df_results,
        'classification_report': report,
        'auc': auc,
        'auprc': auprc,
        'precision_recall': (precision, recall),
        'confusion_matrix': cm,
        'detection_rate': detection_rate,
        'early_detection_rate': early_detection_rate,
        'avg_lead_time': avg_lead_time,
        'false_alarm_rate': false_alarm_rate,
        'patient_predictions': patient_predictions,
        'all_predictions': all_preds,
        'all_targets': all_targets
    }

#################################################
# Visualization Functions
#################################################

def plot_training_history(history):
    """
    Plot training history
    
    Args:
        history: Training history dictionary
        
    Returns:
        matplotlib.figure.Figure: Training history plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot loss
    axes[0].plot(history['train_loss'], label='Train Loss')
    axes[0].plot(history['val_loss'], label='Validation Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].set_title('Training and Validation Loss')
    axes[0].grid(True, alpha=0.3)
    
    # Plot AUC
    axes[1].plot(history['train_auc'], label='Train AUC')
    axes[1].plot(history['val_auc'], label='Validation AUC')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('AUC')
    axes[1].legend()
    axes[1].set_title('Training and Validation AUC')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    return fig

def plot_roc_curve(all_targets, all_predictions):
    """
    Plot ROC curve
    
    Args:
        all_targets: All target values
        all_predictions: All predicted probabilities
        
    Returns:
        matplotlib.figure.Figure: ROC curve plot
    """
    from sklearn.metrics import roc_curve
    
    fpr, tpr, thresholds = roc_curve(all_targets, all_predictions)
    auc = roc_auc_score(all_targets, all_predictions)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(fpr, tpr, label=f'AUC = {auc:.3f}')
    ax.plot([0, 1], [0, 1], 'k--')  # Diagonal line
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig

def plot_precision_recall_curve(precision, recall, auprc):
    """
    Plot Precision-Recall curve
    
    Args:
        precision: Precision values
        recall: Recall values
        auprc: Area under Precision-Recall curve
        
    Returns:
        matplotlib.figure.Figure: Precision-Recall curve plot
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(recall, precision, label=f'AUPRC = {auprc:.3f}')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig

def plot_confusion_matrix(cm):
    """
    Plot confusion matrix
    
    Args:
        cm: Confusion matrix
        
    Returns:
        matplotlib.figure.Figure: Confusion matrix plot
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=['No Sepsis', 'Sepsis'],
        yticklabels=['No Sepsis', 'Sepsis']
    )
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix')
    
    return fig

def plot_thresholds(threshold_results):
    """
    Plot metrics at different thresholds
    
    Args:
        threshold_results: Dataframe with threshold results
        
    Returns:
        matplotlib.figure.Figure: Threshold plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(threshold_results['threshold'], threshold_results['f1'], 'o-', label='F1 Score')
    ax.plot(threshold_results['threshold'], threshold_results['precision'], 's-', label='Precision')
    ax.plot(threshold_results['threshold'], threshold_results['recall'], '^-', label='Recall')
    
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Score')
    ax.set_title('Metrics at Different Thresholds')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Find optimal F1 threshold
    optimal_idx = threshold_results['f1'].idxmax()
    optimal_threshold = threshold_results.loc[optimal_idx, 'threshold']
    optimal_f1 = threshold_results.loc[optimal_idx, 'f1']
    
    # Add vertical line at optimal threshold
    ax.axvline(x=optimal_threshold, color='r', linestyle='--', alpha=0.5)
    ax.text(
        optimal_threshold, 
        0.1, 
        f'Optimal F1: {optimal_f1:.3f}\nThreshold: {optimal_threshold:.1f}', 
        rotation=90, 
        va='bottom'
    )
    
    plt.tight_layout()
    
    return fig

def plot_patient_timeline(patient_id, predictions, targets, threshold=0.5):
    """
    Plot sepsis prediction timeline for a single patient
    
    Args:
        patient_id: Patient identifier
        predictions: Predicted probabilities for each time step
        targets: Actual sepsis labels for each time step
        threshold: Classification threshold
    
    Returns:
        matplotlib.figure.Figure: Timeline visualization
    """
    # Find sepsis onset if any
    sepsis_indices = np.where(targets > 0)[0]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot timeline
    hours = np.arange(len(predictions))
    ax.plot(hours, predictions, 'b-', linewidth=2, label='Sepsis Risk')
    
    # Plot decision threshold
    ax.axhline(y=threshold, color='r', linestyle='--', linewidth=1.5, label=f'Threshold ({threshold:.2f})')
    
    # Determine alert times (predictions above threshold)
    alerts = predictions > threshold
    
    # Mark the alerts
    alert_indices = np.where(alerts)[0]
    if len(alert_indices) > 0:
        first_alert = alert_indices[0]
        ax.axvline(x=first_alert, color='orange', linestyle=':', linewidth=1.5, label='First Alert')
        
        # Mark all alerts
        ax.fill_between(hours, 0, 1, where=alerts, color='orange', alpha=0.3)
    
    # Mark actual sepsis onset if present
    if len(sepsis_indices) > 0:
        sepsis_onset = sepsis_indices[0]
        ax.axvline(x=sepsis_onset, color='red', linewidth=2, label='Sepsis Onset')
        
        # Calculate prediction timing
        if len(alert_indices) > 0:
            lead_time = sepsis_onset - first_alert
            if lead_time > 0:
                ax.set_title(f'Patient {patient_id}: Early Sepsis Detection ({lead_time}h before onset)')
            else:
                ax.set_title(f'Patient {patient_id}: Late Sepsis Detection ({-lead_time}h after onset)')
        else:
            ax.set_title(f'Patient {patient_id}: Missed Sepsis Case')
    else:
        if len(alert_indices) > 0:
            ax.set_title(f'Patient {patient_id}: False Alarm')
        else:
            ax.set_title(f'Patient {patient_id}: True Negative (No Sepsis)')
    
    # Set labels and grid
    ax.set_xlabel('Hours in ICU')
    ax.set_ylabel('Sepsis Risk')
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left')
    
    return fig

###################
# Main Function
####################

def run_sepsis_prediction(
    # Data parameters
    max_seq_length=48,  # Maximum sequence length in hours
    output_dir="./sepsis_results",  # Output directory for results
    subset_size=5000,  # Total number of patients to use
    
    # Split parameters
    test_size=0.2,  # Test set ratio
    val_size=0.25,  # Validation set ratio from training
    
    # Training parameters
    batch_size=8,  # Batch size for training
    epochs=30,  # Number of training epochs
    learning_rate=0.001,  # Learning rate
    seed=42,  # Random seed for reproducibility
    
):
    """
    Main function to run the time step-level sepsis prediction pipeline
    
    Args:
        max_seq_length (int): Maximum sequence length in hours
        output_dir (str): Output directory for results
        test_size (float): Test set ratio
        val_size (float): Validation set ratio (from train set)
        batch_size (int): Batch size for training
        epochs (int): Number of training epochs
        learning_rate (float): Learning rate
        seed (int): Random seed for reproducibility
        
    Returns:
        tuple: (model, evaluation_results)
    """
    # Set random seeds for reproducibility
    set_seeds(seed)
    
    # Set device for PyTorch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")
    
    # Load all data
    patients, patient_ids, dataset_sources = load_patient_data()
    
    # Sample a balanced subset if requested
    if subset_size > 0 and subset_size < len(patients):
        logger.info(f"Sampling balanced subset of {subset_size} patients from original {len(patients)}")
        patients, patient_ids = sample_balanced_dataset(
            patients, patient_ids, dataset_sources, subset_size
        )
    
    # Add engineered features
    patients = add_time_features(patients)
    patients = add_clinical_features(patients)
    
    # Preprocess data
    processed_patients, sepsis_labels, patient_ids, feature_names = preprocess_patient_data(
        patients, max_seq_length=max_seq_length
    )
    
    # Split data by patient
    train_idx, test_idx = train_test_split(
        range(len(processed_patients)), 
        test_size=test_size, 
        random_state=seed,
        stratify=[1 if np.any(labels > 0) else 0 for labels in sepsis_labels]  # Stratify by sepsis/no-sepsis
    )
    train_idx, val_idx = train_test_split(
        train_idx, 
        test_size=val_size, 
        random_state=seed,
        stratify=[1 if np.any(sepsis_labels[i] > 0) else 0 for i in train_idx]  # Stratify by sepsis/no-sepsis
    )
    
    logger.info(f"Data split: {len(train_idx)} train, {len(val_idx)} validation, {len(test_idx)} test")
    
    # Create datasets
    train_dataset = SepsisDataset(
        [processed_patients[i] for i in train_idx],
        [sepsis_labels[i] for i in train_idx]
    )
    
    val_dataset = SepsisDataset(
        [processed_patients[i] for i in val_idx],
        [sepsis_labels[i] for i in val_idx]
    )
    
    test_dataset = SepsisDataset(
        [processed_patients[i] for i in test_idx],
        [sepsis_labels[i] for i in test_idx]
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=1,  # Evaluate one patient at a time for clinical metrics
        shuffle=False, 
        collate_fn=collate_fn
    )
    
    # Calculate class weights to handle imbalance
    pos_count = sum(1 for labels in [sepsis_labels[i] for i in train_idx] for label in labels if label > 0)
    neg_count = sum(1 for labels in [sepsis_labels[i] for i in train_idx] for label in labels if label == 0)
    
    # Determine class weight to address imbalance
    if pos_count > 0:
        pos_weight = neg_count / pos_count
        logger.info(f"Class imbalance - Positive: {pos_count}, Negative: {neg_count}, Weight: {pos_weight:.2f}")
    else:
        pos_weight = 1.0
        logger.warning("No positive examples in training data!")
    
    # Get input dimension
    input_dim = processed_patients[0].shape[1]
    logger.info(f"Input dimension: {input_dim}")
    
    # Train model
    model, history = train_model(
        train_loader, 
        val_loader, 
        input_dim, 
        device,
        class_weight=pos_weight,
        epochs=epochs,
        lr=learning_rate
    )
    
    # Save model
    model_path = os.path.join(output_dir, 'sepsis_timestep_model.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'input_dim': input_dim,
        'feature_names': feature_names,
        'history': history
    }, model_path)
    logger.info(f"Model saved to {model_path}")
    
    # Evaluate model
    patient_ids_test = [patient_ids[i] for i in test_idx]
    evaluation = evaluate_model(model, test_loader, device, patient_ids_test)
    
    # Save evaluation results
    results_path = os.path.join(output_dir, 'timestep_evaluation_results.pkl')
    with open(results_path, 'wb') as f:
        import pickle
        pickle.dump(evaluation, f)
    logger.info(f"Evaluation results saved to {results_path}")
    
    # Generate visualizations
    logger.info("Generating visualizations")
    
    # Training history
    history_fig = plot_training_history(history)
    history_fig.savefig(os.path.join(output_dir, 'training_history.png'))
    
    # ROC and PR curves
    roc_fig = plot_roc_curve(evaluation['all_targets'], evaluation['all_predictions'])
    roc_fig.savefig(os.path.join(output_dir, 'roc_curve.png'))
    
    pr_fig = plot_precision_recall_curve(
        evaluation['precision_recall'][0], 
        evaluation['precision_recall'][1], 
        evaluation['auprc']
    )
    pr_fig.savefig(os.path.join(output_dir, 'precision_recall_curve.png'))
    
    # Generate patient timeline visualizations for a few examples
    patient_metrics = evaluation['patient_results']
    
    # Get a mix of early detections, late detections, and missed cases
    early_detections = patient_metrics[
        (patient_metrics['actual_sepsis'] == 1) & 
        (patient_metrics['prediction_lead_time'] > 0)
    ]
    
    late_detections = patient_metrics[
        (patient_metrics['actual_sepsis'] == 1) & 
        (patient_metrics['detection_delay'] > 0)
    ]
    
    missed_cases = patient_metrics[
        (patient_metrics['actual_sepsis'] == 1) & 
        (patient_metrics['predicted_sepsis'] == 0)
    ]
    
    false_alarms = patient_metrics[
        (patient_metrics['actual_sepsis'] == 0) & 
        (patient_metrics['predicted_sepsis'] == 1)
    ]
    
    # Select example patients from each category
    example_patients = []
    
    # Early detections (up to 3)
    if len(early_detections) > 0:
        # Sort by lead time (highest first)
        sorted_early = early_detections.sort_values('prediction_lead_time', ascending=False)
        example_patients.extend(sorted_early['patient_id'].iloc[:min(3, len(sorted_early))].tolist())
    
    # Late detections (up to 2)
    if len(late_detections) > 0:
        example_patients.extend(late_detections['patient_id'].iloc[:min(2, len(late_detections))].tolist())
    
    # Missed cases (up to 2)
    if len(missed_cases) > 0:
        example_patients.extend(missed_cases['patient_id'].iloc[:min(2, len(missed_cases))].tolist())
    
    # False alarms (up to 2)
    if len(false_alarms) > 0:
        example_patients.extend(false_alarms['patient_id'].iloc[:min(2, len(false_alarms))].tolist())
    
    # Generate and save visualizations for selected patients
    patient_predictions = evaluation['patient_predictions']
    
    for patient_id in example_patients:
        if patient_id in patient_predictions:
            # Generate timeline plot
            fig = plot_patient_timeline(
                patient_id,
                patient_predictions[patient_id]['predictions'],
                patient_predictions[patient_id]['targets'],
                threshold=0.5
            )
            
            # Save figure
            fig.savefig(os.path.join(output_dir, f'patient_{patient_id}_timeline.png'))
            plt.close(fig)
    
    # Print summary results
    logger.info("\n" + "#" * 50)
    logger.info("TIME STEP-LEVEL SEPSIS PREDICTION RESULTS")
    logger.info("#" * 50)
    logger.info(f"AUC: {evaluation['auc']:.4f}")
    logger.info(f"AUPRC: {evaluation['auprc']:.4f}")
    logger.info(f"Detection Rate: {evaluation['detection_rate']:.2f}")
    logger.info(f"Early Detection Rate: {evaluation['early_detection_rate']:.2f}")
    
    if evaluation['avg_lead_time'] > 0:
        logger.info(f"Average Lead Time: {evaluation['avg_lead_time']:.2f} hours")
    
    logger.info(f"False Alarm Rate: {evaluation['false_alarm_rate']:.2f}")
    logger.info("\nClassification Report:")
    logger.info("\n" + evaluation['classification_report'])
    logger.info("#" * 50)
    
    # Show all figures
    plt.show()
    
    return model, evaluation

#################################################
# execution Block
#################################################

if __name__ == "__main__":
    # Run with balanced dataset and time step-level prediction
    model, evaluation = run_sepsis_prediction(
        subset_size=5000,         # Use 5000 patients balanced between datasets
        max_seq_length=24,        # Maximum of 24 hours of data per patient
        batch_size=16,            # Adjust based on your GPU memory
        epochs=100                 # Number of training epochs
    )
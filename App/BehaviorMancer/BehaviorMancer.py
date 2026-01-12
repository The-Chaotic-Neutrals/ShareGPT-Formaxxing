"""
BehaviorMancer - Model Behavior Pattern Removal Core Logic

Detects and removes behavior patterns from language models using orthogonal projection.
Can be used to remove any behavior pattern you have training data for:
- Refusal behavior
- Sycophancy
- Verbosity
- Hedging
- Any other pattern with contrastive examples

Based on research from:
- Arditi et al. (2024) - "Refusal in Language Models Is Mediated by a Single Direction"
- Lai - Norm-Preserving Biprojected Abliteration
- Fang et al. (ICLR 2025) - AlphaEdit: Null-Space Constrained Knowledge Editing
"""

import os
import json
import gc
from pathlib import Path
from typing import Callable, Optional
from dataclasses import dataclass

import torch
import numpy as np
from tqdm import tqdm


@dataclass
class BehaviorMancerConfig:
    """Configuration for behavior removal process."""
    
    # Model source
    model_source: str = "local"  # "local" or "huggingface"
    model_path: str = ""  # Local path or HF model ID
    hf_token: Optional[str] = None  # HuggingFace token for gated models
    
    # Target behavior dataset (behavior to EXHIBIT)
    target_source: str = "local"  # "local" or "huggingface"
    target_path: str = ""  # Local path or HF dataset ID
    target_column: str = "text"  # Column name for structured formats (jsonl, parquet, csv)
    
    # Baseline behavior dataset (behavior to REMOVE)
    baseline_source: str = "local"  # "local" or "huggingface"
    baseline_path: str = ""  # Local path or HF dataset ID
    baseline_column: str = "text"  # Column name for structured formats (jsonl, parquet, csv)
    
    # Preservation dataset source (for null-space constraints)
    preservation_source: str = "local"  # "local" or "huggingface"
    preservation_path: str = ""  # Local path or HF dataset ID
    preservation_column: str = "text"  # Column name for structured formats (jsonl, parquet, csv)
    
    # Abliteration settings
    n_samples: int = 30  # Number of sample pairs to use
    direction_multiplier: float = 1.0  # Ablation strength (0.0 - 1.0+)
    precision: str = "float16"  # "float16", "bfloat16", "float32"
    
    # Output
    output_path: str = ""  # Where to save modified model
    
    # Advanced options
    norm_preservation: bool = True  # Preserve weight magnitudes
    winsorization: bool = False  # Clip outlier activations
    winsorization_threshold: float = 0.995  # Quantile threshold for winsorization
    null_space_constraints: bool = False  # Preserve capabilities using preservation dataset
    adaptive_layer_weighting: bool = False  # Focus on middle-to-later layers
    
    # Layer selection
    start_layer_ratio: float = 0.2  # Start from 20% of layers
    end_layer_ratio: float = 0.9  # End at 90% of layers


class BehaviorMancer:
    """Main class for behavior pattern removal."""
    
    def __init__(self, config: Optional[BehaviorMancerConfig] = None, log_callback: Optional[Callable[[str], None]] = None):
        """
        Initialize the behavior mancer.
        
        Args:
            config: Configuration for behavior removal
            log_callback: Optional callback for logging messages
        """
        self.config = config or BehaviorMancerConfig()
        self.log_callback = log_callback or print
        
        self.model = None
        self.tokenizer = None
        self.behavior_direction = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Stop flag for interruption
        self.stop_requested = False
    
    def log(self, message: str):
        """Log a message."""
        if self.log_callback:
            self.log_callback(message)
    
    def request_stop(self):
        """Request the process to stop."""
        self.stop_requested = True
    
    def _get_dtype(self) -> torch.dtype:
        """Get the torch dtype based on config."""
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        return dtype_map.get(self.config.precision, torch.float16)
    
    def _load_local_dataset(self, file_path: str, column: str = "text") -> list[str]:
        """
        Load a dataset from a local file. Supports multiple formats:
        - .txt: Plain text, one sample per line (lines starting with # are ignored)
        - .jsonl: JSON Lines, extracts specified column
        - .json: JSON array, extracts specified column
        - .parquet: Parquet file, extracts specified column
        - .csv: CSV file, extracts specified column
        
        Args:
            file_path: Path to the file
            column: Column name for structured formats (ignored for .txt)
            
        Returns:
            List of text samples
        """
        samples = []
        ext = Path(file_path).suffix.lower()
        
        if ext == ".txt":
            # Plain text: one sample per line, ignore comments
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        samples.append(line)
        
        elif ext == ".jsonl":
            # JSON Lines format
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        if isinstance(data, dict) and column in data:
                            samples.append(str(data[column]))
                        elif isinstance(data, str):
                            samples.append(data)
                    except json.JSONDecodeError:
                        continue
        
        elif ext == ".json":
            # JSON array format
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict) and column in item:
                            samples.append(str(item[column]))
                        elif isinstance(item, str):
                            samples.append(item)
        
        elif ext == ".parquet":
            # Parquet format
            try:
                import pandas as pd
                df = pd.read_parquet(file_path)
                if column in df.columns:
                    samples = df[column].dropna().astype(str).tolist()
                elif len(df.columns) == 1:
                    # Single column, use it
                    samples = df.iloc[:, 0].dropna().astype(str).tolist()
                else:
                    self.log(f"Warning: Column '{column}' not found in parquet. Available: {list(df.columns)}")
            except ImportError:
                self.log("Error: pandas/pyarrow required for parquet files. Install with: pip install pandas pyarrow")
        
        elif ext == ".csv":
            # CSV format
            try:
                import pandas as pd
                df = pd.read_csv(file_path)
                if column in df.columns:
                    samples = df[column].dropna().astype(str).tolist()
                elif len(df.columns) == 1:
                    # Single column, use it
                    samples = df.iloc[:, 0].dropna().astype(str).tolist()
                else:
                    self.log(f"Warning: Column '{column}' not found in CSV. Available: {list(df.columns)}")
            except ImportError:
                self.log("Error: pandas required for CSV files. Install with: pip install pandas")
        
        else:
            self.log(f"Warning: Unknown file format '{ext}'. Trying to read as plain text.")
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        samples.append(line)
        
        return samples
    
    def _load_hf_dataset(self, dataset_id: str, column: str = "text") -> list[str]:
        """
        Load a dataset from HuggingFace Hub.
        
        Args:
            dataset_id: HuggingFace dataset ID
            column: Column name to extract
            
        Returns:
            List of text samples
        """
        from datasets import load_dataset
        
        kwargs = {}
        if self.config.hf_token:
            kwargs["token"] = self.config.hf_token
        
        dataset = load_dataset(dataset_id, **kwargs)
        
        # Get the train split or first available split
        if "train" in dataset:
            data = dataset["train"]
        else:
            data = dataset[list(dataset.keys())[0]]
        
        if column in data.column_names:
            return list(data[column])
        elif len(data.column_names) == 1:
            return list(data[data.column_names[0]])
        else:
            self.log(f"Warning: Column '{column}' not found. Available: {data.column_names}")
            return []
    
    def load_model(self) -> bool:
        """
        Load the model from local path or HuggingFace.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            model_path = self.config.model_path
            
            if self.config.model_source == "huggingface":
                self.log(f"Loading model from HuggingFace: {model_path}")
                kwargs = {
                    "torch_dtype": self._get_dtype(),
                    "device_map": "auto",
                    "trust_remote_code": True,
                }
                if self.config.hf_token:
                    kwargs["token"] = self.config.hf_token
                
                self.model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_path, 
                    token=self.config.hf_token,
                    trust_remote_code=True
                )
            else:
                self.log(f"Loading model from local path: {model_path}")
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=self._get_dtype(),
                    device_map="auto",
                    trust_remote_code=True,
                    local_files_only=True,
                )
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_path,
                    trust_remote_code=True,
                    local_files_only=True,
                )
            
            # Ensure tokenizer has pad token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.log(f"Model loaded successfully. Layers: {self.model.config.num_hidden_layers}")
            return True
            
        except Exception as e:
            self.log(f"Error loading model: {str(e)}")
            return False
    
    def load_behavior_dataset(self) -> tuple[list[str], list[str]]:
        """
        Load behavior datasets from configured sources.
        
        Target and baseline datasets are loaded separately, supporting multiple formats:
        - .txt: Plain text, one sample per line
        - .jsonl: JSON Lines format
        - .json: JSON array format
        - .parquet: Parquet format
        - .csv: CSV format
        
        Returns:
            Tuple of (target_behavior_samples, baseline_behavior_samples)
        """
        # Load target behavior dataset (behavior to EXHIBIT)
        if self.config.target_source == "local":
            self.log(f"Loading target behavior from local file: {self.config.target_path}")
            target_samples = self._load_local_dataset(self.config.target_path, self.config.target_column)
        elif self.config.target_source == "huggingface":
            self.log(f"Loading target behavior from HuggingFace: {self.config.target_path}")
            target_samples = self._load_hf_dataset(self.config.target_path, self.config.target_column)
        else:
            raise ValueError(f"Unknown target source: {self.config.target_source}")
        
        # Load baseline behavior dataset (behavior to REMOVE)
        if self.config.baseline_source == "local":
            self.log(f"Loading baseline behavior from local file: {self.config.baseline_path}")
            baseline_samples = self._load_local_dataset(self.config.baseline_path, self.config.baseline_column)
        elif self.config.baseline_source == "huggingface":
            self.log(f"Loading baseline behavior from HuggingFace: {self.config.baseline_path}")
            baseline_samples = self._load_hf_dataset(self.config.baseline_path, self.config.baseline_column)
        else:
            raise ValueError(f"Unknown baseline source: {self.config.baseline_source}")
        
        self.log(f"Loaded {len(target_samples)} target and {len(baseline_samples)} baseline samples")
        return target_samples, baseline_samples
    
    def load_preservation_dataset(self) -> list[str]:
        """
        Load preservation dataset for null-space constraints.
        
        These are prompts that test capabilities you want to preserve
        (e.g., math, coding, reasoning, general knowledge).
        
        Supports multiple formats:
        - .txt: Plain text, one sample per line
        - .jsonl: JSON Lines format
        - .json: JSON array format
        - .parquet: Parquet format
        - .csv: CSV format
        
        Returns:
            List of preservation prompts
        """
        if not self.config.preservation_path:
            self.log("Warning: No preservation dataset specified for null-space constraints")
            return []
        
        if self.config.preservation_source == "local":
            self.log(f"Loading preservation dataset from local file: {self.config.preservation_path}")
            prompts = self._load_local_dataset(self.config.preservation_path, self.config.preservation_column)
        elif self.config.preservation_source == "huggingface":
            self.log(f"Loading preservation dataset from HuggingFace: {self.config.preservation_path}")
            prompts = self._load_hf_dataset(self.config.preservation_path, self.config.preservation_column)
        else:
            raise ValueError(f"Unknown preservation source: {self.config.preservation_source}")
        
        self.log(f"Loaded {len(prompts)} preservation prompts")
        return prompts
    
    def _get_hidden_states(self, prompts: list[str], layer_indices: list[int]) -> dict[int, torch.Tensor]:
        """
        Extract hidden states from the model for given prompts.
        
        Args:
            prompts: List of prompts to process
            layer_indices: Which layers to extract from
            
        Returns:
            Dictionary mapping layer index to stacked hidden states
        """
        hidden_states = {i: [] for i in layer_indices}
        
        self.model.eval()
        
        with torch.no_grad():
            for prompt in tqdm(prompts, desc="Extracting hidden states", leave=False):
                if self.stop_requested:
                    break
                
                inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
                
                outputs = self.model(**inputs, output_hidden_states=True)
                
                for layer_idx in layer_indices:
                    # Get the last token's hidden state from this layer
                    layer_hidden = outputs.hidden_states[layer_idx][:, -1, :].cpu()
                    hidden_states[layer_idx].append(layer_hidden)
        
        # Stack all hidden states per layer
        for layer_idx in layer_indices:
            if hidden_states[layer_idx]:
                hidden_states[layer_idx] = torch.cat(hidden_states[layer_idx], dim=0)
            else:
                hidden_states[layer_idx] = torch.tensor([])
        
        return hidden_states
    
    def _winsorize(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Apply winsorization to clip outlier activations.
        
        Args:
            tensor: Input tensor
            
        Returns:
            Winsorized tensor
        """
        if not self.config.winsorization:
            return tensor
        
        threshold = torch.quantile(tensor.abs().float(), self.config.winsorization_threshold)
        return torch.clamp(tensor, -threshold, threshold)
    
    def extract_behavior_direction(self, target_samples: list[str], baseline_samples: list[str]) -> torch.Tensor:
        """
        Extract the behavior direction from model activations.
        
        The behavior direction is computed as:
        d = normalize(mean(baseline_activations) - mean(target_activations))
        
        Args:
            target_samples: Samples exhibiting the desired behavior
            baseline_samples: Samples exhibiting behavior to remove
            
        Returns:
            Normalized behavior direction vector
        """
        n_layers = self.model.config.num_hidden_layers
        
        # Determine which layers to extract from (middle layers are most important)
        start_layer = int(n_layers * self.config.start_layer_ratio)
        end_layer = int(n_layers * self.config.end_layer_ratio)
        layer_indices = list(range(start_layer, end_layer + 1))
        
        self.log(f"Extracting hidden states from layers {start_layer} to {end_layer}")
        
        # Limit samples
        n = min(self.config.n_samples, len(target_samples), len(baseline_samples))
        target_samples = target_samples[:n]
        baseline_samples = baseline_samples[:n]
        
        self.log(f"Using {n} sample pairs")
        
        # Extract hidden states
        self.log("Processing target behavior samples (to exhibit)...")
        target_states = self._get_hidden_states(target_samples, layer_indices)
        
        if self.stop_requested:
            return None
        
        self.log("Processing baseline behavior samples (to remove)...")
        baseline_states = self._get_hidden_states(baseline_samples, layer_indices)
        
        if self.stop_requested:
            return None
        
        # Compute behavior direction for each layer and average
        directions = []
        
        for layer_idx in layer_indices:
            target = target_states[layer_idx]
            baseline = baseline_states[layer_idx]
            
            if len(target) == 0 or len(baseline) == 0:
                continue
            
            # Apply winsorization if enabled
            target = self._winsorize(target)
            baseline = self._winsorize(baseline)
            
            # Compute mean difference (baseline has behavior to remove, target has behavior to exhibit)
            diff = baseline.mean(dim=0) - target.mean(dim=0)
            
            # Normalize
            diff = diff / (diff.norm() + 1e-8)
            directions.append(diff)
        
        if not directions:
            raise ValueError("No valid directions computed")
        
        # Average directions across layers
        behavior_direction = torch.stack(directions).mean(dim=0)
        behavior_direction = behavior_direction / (behavior_direction.norm() + 1e-8)
        
        self.behavior_direction = behavior_direction
        self.log(f"Behavior direction extracted (shape: {behavior_direction.shape})")
        
        return behavior_direction
    
    def compute_null_space_projector(self, preservation_prompts: list[str]) -> Optional[torch.Tensor]:
        """
        Compute null-space projector from preservation activations.
        
        Based on AlphaEdit (Fang et al., ICLR 2025):
        1. Collect activations K from preservation prompts
        2. Compute SVD: U, S, V = SVD(K)
        3. Build null-space projector: P_null = I - V @ V.T
        
        The projector ensures weight updates don't affect outputs for preserved capabilities.
        
        Args:
            preservation_prompts: Prompts testing capabilities to preserve
            
        Returns:
            Null-space projection matrix or None if computation fails
        """
        if not preservation_prompts:
            self.log("No preservation prompts provided, skipping null-space computation")
            return None
        
        n_layers = self.model.config.num_hidden_layers
        start_layer = int(n_layers * self.config.start_layer_ratio)
        end_layer = int(n_layers * self.config.end_layer_ratio)
        layer_indices = list(range(start_layer, end_layer + 1))
        
        self.log(f"Computing null-space projector from {len(preservation_prompts)} prompts...")
        
        # Get hidden states for preservation prompts
        preservation_states = self._get_hidden_states(preservation_prompts, layer_indices)
        
        if self.stop_requested:
            return None
        
        # Collect all activations across layers
        all_activations = []
        for layer_idx in layer_indices:
            if len(preservation_states[layer_idx]) > 0:
                all_activations.append(preservation_states[layer_idx])
        
        if not all_activations:
            self.log("Warning: No valid activations for null-space computation")
            return None
        
        # Stack activations: shape (n_prompts * n_layers, hidden_dim)
        K = torch.cat(all_activations, dim=0).float()
        
        self.log(f"Activation matrix shape: {K.shape}")
        
        # Compute SVD
        try:
            # Use truncated SVD for efficiency
            # Keep top singular vectors that capture most variance
            U, S, Vh = torch.linalg.svd(K, full_matrices=False)
            
            # Keep singular vectors with significant singular values
            # (threshold at 1% of max singular value)
            threshold = 0.01 * S[0]
            n_keep = (S > threshold).sum().item()
            n_keep = max(n_keep, 1)  # Keep at least 1
            
            self.log(f"Keeping {n_keep} singular vectors for null-space projector")
            
            # V contains right singular vectors (hidden_dim x min(n_samples, hidden_dim))
            V = Vh[:n_keep, :].T  # shape: (hidden_dim, n_keep)
            
            # Null-space projector: P_null = I - V @ V.T
            hidden_dim = K.shape[1]
            I = torch.eye(hidden_dim, dtype=torch.float32)
            P_null = I - V @ V.T
            
            self.log(f"Null-space projector computed (shape: {P_null.shape})")
            
            return P_null
            
        except Exception as e:
            self.log(f"Warning: SVD computation failed: {str(e)}")
            return None
    
    def _get_layer_weight(self, layer_idx: int, n_layers: int) -> float:
        """
        Get the ablation weight for a specific layer.
        
        Uses Gaussian weighting centered at 60% of model depth.
        
        Args:
            layer_idx: Layer index
            n_layers: Total number of layers
            
        Returns:
            Weight multiplier for this layer
        """
        if not self.config.adaptive_layer_weighting:
            return 1.0
        
        # Gaussian centered at 60% of depth, with 20% std
        mu = 0.6 * n_layers
        sigma = 0.2 * n_layers
        
        weight = np.exp(-0.5 * ((layer_idx - mu) / sigma) ** 2)
        return float(weight)
    
    def _project_out_direction(self, weight: torch.Tensor, direction: torch.Tensor, strength: float = 1.0) -> torch.Tensor:
        """
        Project out the behavior direction from a weight matrix.
        
        W_proj = W - strength * (W @ d) @ d.T
        
        Args:
            weight: Weight matrix to modify
            direction: Behavior direction to project out
            strength: Ablation strength multiplier
            
        Returns:
            Modified weight matrix
        """
        # Ensure direction is on same device and dtype
        d = direction.to(weight.device, weight.dtype)
        
        # Ensure direction is a column vector
        if d.dim() == 1:
            d = d.unsqueeze(1)
        
        # Handle shape mismatches
        if weight.shape[-1] != d.shape[0]:
            # Direction might need to be applied differently based on weight shape
            if weight.shape[0] == d.shape[0]:
                # Apply to rows instead
                projection = (weight @ d) @ d.T
                projection = projection.T
            else:
                self.log(f"Warning: Shape mismatch - weight {weight.shape}, direction {d.shape}")
                return weight
        else:
            # Standard case: project out from columns
            projection = (weight @ d) @ d.T
        
        # Apply projection
        weight_proj = weight - strength * projection
        
        return weight_proj
    
    def _preserve_norm(self, original: torch.Tensor, modified: torch.Tensor) -> torch.Tensor:
        """
        Rescale modified weights to preserve original Frobenius norm.
        
        Args:
            original: Original weight matrix
            modified: Modified weight matrix
            
        Returns:
            Rescaled weight matrix
        """
        if not self.config.norm_preservation:
            return modified
        
        original_norm = original.norm()
        modified_norm = modified.norm()
        
        if modified_norm > 0:
            return modified * (original_norm / modified_norm)
        return modified
    
    def remove_behavior(self, null_space_projector: Optional[torch.Tensor] = None) -> bool:
        """
        Perform the behavior removal process.
        
        Args:
            null_space_projector: Optional null-space projector for capability preservation
        
        Returns:
            True if successful, False otherwise
        """
        if self.model is None:
            self.log("Error: Model not loaded")
            return False
        
        if self.behavior_direction is None:
            self.log("Error: Behavior direction not extracted")
            return False
        
        self.log("Starting behavior removal process...")
        
        n_layers = self.model.config.num_hidden_layers
        direction = self.behavior_direction
        strength = self.config.direction_multiplier
        
        # Determine which layers to modify
        start_layer = int(n_layers * self.config.start_layer_ratio)
        end_layer = int(n_layers * self.config.end_layer_ratio)
        
        self.log(f"Modifying layers {start_layer} to {end_layer} with strength {strength}")
        
        if null_space_projector is not None:
            self.log("Applying null-space constraints for capability preservation")
        
        # Get the model's layer modules
        # This works for most transformer architectures
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            layers = self.model.model.layers
        elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
            layers = self.model.transformer.h
        elif hasattr(self.model, 'gpt_neox') and hasattr(self.model.gpt_neox, 'layers'):
            layers = self.model.gpt_neox.layers
        else:
            self.log("Error: Could not find model layers. Unsupported architecture.")
            return False
        
        modified_count = 0
        
        for layer_idx in tqdm(range(start_layer, end_layer + 1), desc="Removing behavior pattern"):
            if self.stop_requested:
                self.log("Process stopped by user")
                return False
            
            layer = layers[layer_idx]
            layer_weight = self._get_layer_weight(layer_idx, n_layers)
            effective_strength = strength * layer_weight
            
            # Find and modify attention and MLP weights
            for name, param in layer.named_parameters():
                if not param.requires_grad:
                    continue
                
                # Target projection layers (q, k, v, o projections and MLP)
                target_names = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 
                               'gate_proj', 'up_proj', 'down_proj',
                               'query', 'key', 'value', 'dense',
                               'c_attn', 'c_proj', 'c_fc']
                
                if any(target in name for target in target_names):
                    with torch.no_grad():
                        original_weight = param.data.clone()
                        
                        # Project out behavior direction
                        modified_weight = self._project_out_direction(
                            param.data, direction, effective_strength
                        )
                        
                        # Apply null-space constraint if provided
                        if null_space_projector is not None:
                            # Compute the weight delta
                            delta = modified_weight - original_weight
                            
                            # Project delta into null-space of preservation activations
                            # This ensures the update doesn't affect preserved capabilities
                            P_null = null_space_projector.to(delta.device, delta.dtype)
                            
                            # Apply null-space projection to rows or columns based on shape
                            if delta.shape[-1] == P_null.shape[0]:
                                delta_constrained = delta @ P_null
                            elif delta.shape[0] == P_null.shape[0]:
                                delta_constrained = P_null @ delta
                            else:
                                # Shape mismatch, skip null-space for this weight
                                delta_constrained = delta
                            
                            modified_weight = original_weight + delta_constrained
                        
                        # Preserve norm
                        modified_weight = self._preserve_norm(original_weight, modified_weight)
                        
                        param.data = modified_weight
                        modified_count += 1
        
        self.log(f"Behavior removal complete. Modified {modified_count} weight matrices.")
        return True
    
    def save_model(self) -> bool:
        """
        Save the modified model.
        
        Returns:
            True if successful, False otherwise
        """
        if self.model is None:
            self.log("Error: No model to save")
            return False
        
        output_path = self.config.output_path
        if not output_path:
            self.log("Error: No output path specified")
            return False
        
        try:
            os.makedirs(output_path, exist_ok=True)
            
            self.log(f"Saving modified model to: {output_path}")
            
            # Save model
            self.model.save_pretrained(output_path, safe_serialization=True)
            
            # Save tokenizer
            self.tokenizer.save_pretrained(output_path)
            
            # Save config
            config_path = os.path.join(output_path, "behaviormancer_config.json")
            with open(config_path, 'w', encoding='utf-8') as f:
                config_dict = {
                    "model_source": self.config.model_source,
                    "model_path": self.config.model_path,
                    "target_source": self.config.target_source,
                    "target_path": self.config.target_path,
                    "target_column": self.config.target_column,
                    "baseline_source": self.config.baseline_source,
                    "baseline_path": self.config.baseline_path,
                    "baseline_column": self.config.baseline_column,
                    "preservation_source": self.config.preservation_source,
                    "preservation_path": self.config.preservation_path,
                    "preservation_column": self.config.preservation_column,
                    "n_samples": self.config.n_samples,
                    "direction_multiplier": self.config.direction_multiplier,
                    "precision": self.config.precision,
                    "norm_preservation": self.config.norm_preservation,
                    "winsorization": self.config.winsorization,
                    "null_space_constraints": self.config.null_space_constraints,
                    "adaptive_layer_weighting": self.config.adaptive_layer_weighting,
                }
                json.dump(config_dict, f, indent=2)
            
            self.log("Model saved successfully!")
            return True
            
        except Exception as e:
            self.log(f"Error saving model: {str(e)}")
            return False
    
    def test_model(self, prompt: str, max_new_tokens: int = 256) -> str:
        """
        Test the model with a prompt.
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            
        Returns:
            Generated response
        """
        if self.model is None or self.tokenizer is None:
            return "Error: Model not loaded"
        
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Remove the input prompt from response
            if response.startswith(prompt):
                response = response[len(prompt):].strip()
            
            return response
            
        except Exception as e:
            return f"Error during generation: {str(e)}"
    
    def get_model_info(self) -> dict:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model information
        """
        if self.model is None:
            return {"error": "No model loaded"}
        
        config = self.model.config
        
        info = {
            "architecture": config.architectures[0] if hasattr(config, 'architectures') and config.architectures else "Unknown",
            "hidden_size": getattr(config, 'hidden_size', 'Unknown'),
            "num_layers": getattr(config, 'num_hidden_layers', 'Unknown'),
            "num_attention_heads": getattr(config, 'num_attention_heads', 'Unknown'),
            "vocab_size": getattr(config, 'vocab_size', 'Unknown'),
            "max_position_embeddings": getattr(config, 'max_position_embeddings', 'Unknown'),
        }
        
        # Calculate approximate parameter count
        total_params = sum(p.numel() for p in self.model.parameters())
        info["parameters"] = f"{total_params / 1e9:.2f}B" if total_params >= 1e9 else f"{total_params / 1e6:.2f}M"
        
        return info
    
    def cleanup(self):
        """Clean up resources."""
        if self.model is not None:
            del self.model
            self.model = None
        
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        
        self.behavior_direction = None
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self.log("Resources cleaned up")
    
    def run_full_pipeline(self) -> bool:
        """
        Run the complete behavior removal pipeline.
        
        Returns:
            True if successful, False otherwise
        """
        self.stop_requested = False
        
        # Determine total steps based on whether null-space constraints are enabled
        total_steps = 6 if self.config.null_space_constraints else 5
        step = 0
        
        try:
            # Step 1: Load model
            step += 1
            self.log(f"Step {step}/{total_steps}: Loading model...")
            if not self.load_model():
                return False
            
            if self.stop_requested:
                return False
            
            # Step 2: Load behavior dataset
            step += 1
            self.log(f"Step {step}/{total_steps}: Loading behavior dataset...")
            target_samples, baseline_samples = self.load_behavior_dataset()
            
            if self.stop_requested:
                return False
            
            # Step 3: Extract behavior direction
            step += 1
            self.log(f"Step {step}/{total_steps}: Extracting behavior direction...")
            self.extract_behavior_direction(target_samples, baseline_samples)
            
            if self.stop_requested:
                return False
            
            # Step 3.5 (optional): Compute null-space projector
            null_space_projector = None
            if self.config.null_space_constraints:
                step += 1
                self.log(f"Step {step}/{total_steps}: Computing null-space projector...")
                preservation_prompts = self.load_preservation_dataset()
                
                if preservation_prompts:
                    null_space_projector = self.compute_null_space_projector(preservation_prompts)
                    
                    if null_space_projector is None:
                        self.log("Warning: Could not compute null-space projector, proceeding without constraints")
                else:
                    self.log("Warning: No preservation prompts loaded, disabling null-space constraints")
            
            if self.stop_requested:
                return False
            
            # Step 4: Remove behavior
            step += 1
            self.log(f"Step {step}/{total_steps}: Removing behavior pattern...")
            if not self.remove_behavior(null_space_projector):
                return False
            
            if self.stop_requested:
                return False
            
            # Step 5: Save model
            step += 1
            self.log(f"Step {step}/{total_steps}: Saving modified model...")
            if not self.save_model():
                return False
            
            self.log("Pipeline completed successfully!")
            return True
            
        except Exception as e:
            self.log(f"Pipeline failed: {str(e)}")
            import traceback
            self.log(traceback.format_exc())
            return False

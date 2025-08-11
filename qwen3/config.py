"""
Unified configuration management for Qwen3.

Supports:
- Model configurations
- Inference configurations  
- Training configurations
- Deployment configurations
"""

import os
import yaml
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Union, List
from pathlib import Path


@dataclass
class ModelConfig:
    """Model configuration."""
    model_name_or_path: str
    model_type: str = "qwen3"
    torch_dtype: str = "auto"
    device_map: str = "auto"
    trust_remote_code: bool = True
    use_flash_attention: bool = True
    max_model_len: Optional[int] = None
    
    
@dataclass 
class InferenceConfig:
    """Inference configuration."""
    max_new_tokens: int = 2048
    temperature: float = 0.7
    top_p: float = 0.8
    top_k: int = 20
    repetition_penalty: float = 1.05
    do_sample: bool = True
    stream: bool = True
    enable_thinking: Optional[bool] = None


@dataclass
class TrainingConfig:
    """Training configuration."""
    output_dir: str
    learning_rate: float = 5e-5
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 1
    warmup_steps: int = 100
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    dataloader_num_workers: int = 4
    remove_unused_columns: bool = False
    
    # LoRA specific
    use_lora: bool = False
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    
    # QLoRA specific  
    use_qlora: bool = False
    bits: int = 4
    quant_type: str = "nf4"
    double_quant: bool = True


@dataclass
class DeploymentConfig:
    """Deployment configuration."""
    framework: str = "vllm"  # vllm, sglang, tensorrt-llm, transformers
    host: str = "0.0.0.0"
    port: int = 8000
    num_gpus: int = 1
    max_num_seqs: int = 256
    max_model_len: int = 32768
    gpu_memory_utilization: float = 0.9
    enable_prefix_caching: bool = True
    disable_log_requests: bool = False
    
    # Docker specific
    use_docker: bool = False
    docker_image: str = "qwen3:latest"
    docker_ports: Dict[str, int] = field(default_factory=lambda: {"8000/tcp": 8000})


class ConfigManager:
    """Centralized configuration manager."""
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        self.config_path = Path(config_path) if config_path else None
        self._configs: Dict[str, Any] = {}
        
        if self.config_path and self.config_path.exists():
            self.load_from_file(self.config_path)
    
    def load_from_file(self, config_path: Union[str, Path]) -> None:
        """Load configuration from YAML file."""
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
            
        with open(config_path, 'r', encoding='utf-8') as f:
            self._configs = yaml.safe_load(f)
    
    def save_to_file(self, config_path: Union[str, Path]) -> None:
        """Save configuration to YAML file."""
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(self._configs, f, default_flow_style=False, allow_unicode=True)
    
    def get_model_config(self, **kwargs) -> ModelConfig:
        """Get model configuration."""
        config_dict = self._configs.get('model', {})
        config_dict.update(kwargs)
        return ModelConfig(**config_dict)
    
    def get_inference_config(self, **kwargs) -> InferenceConfig:
        """Get inference configuration."""
        config_dict = self._configs.get('inference', {})
        config_dict.update(kwargs)
        return InferenceConfig(**config_dict)
    
    def get_training_config(self, **kwargs) -> TrainingConfig:
        """Get training configuration."""
        config_dict = self._configs.get('training', {})
        config_dict.update(kwargs)
        return TrainingConfig(**config_dict)
    
    def get_deployment_config(self, **kwargs) -> DeploymentConfig:
        """Get deployment configuration.""" 
        config_dict = self._configs.get('deployment', {})
        config_dict.update(kwargs)
        return DeploymentConfig(**config_dict)
    
    def update_config(self, section: str, **kwargs) -> None:
        """Update configuration section."""
        if section not in self._configs:
            self._configs[section] = {}
        self._configs[section].update(kwargs)
    
    @classmethod
    def from_env(cls) -> 'ConfigManager':
        """Create configuration manager from environment variables."""
        config_path = os.getenv('QWEN3_CONFIG_PATH')
        return cls(config_path)


# Global configuration manager instance
config_manager = ConfigManager.from_env()
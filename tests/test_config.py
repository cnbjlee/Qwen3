"""
Unit tests for configuration management.
"""

import os
import tempfile
import pytest
import yaml
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent))

from qwen3.config import (
    ModelConfig,
    InferenceConfig,
    TrainingConfig,
    DeploymentConfig,
    ConfigManager
)


class TestModelConfig:
    """Test ModelConfig class."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = ModelConfig(model_name_or_path="test_model")
        
        assert config.model_name_or_path == "test_model"
        assert config.model_type == "qwen3"
        assert config.torch_dtype == "auto"
        assert config.device_map == "auto"
        assert config.trust_remote_code is True
        assert config.use_flash_attention is True
        assert config.max_model_len is None
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = ModelConfig(
            model_name_or_path="custom_model",
            torch_dtype="float16",
            device_map="cpu",
            max_model_len=4096
        )
        
        assert config.model_name_or_path == "custom_model"
        assert config.torch_dtype == "float16"
        assert config.device_map == "cpu"
        assert config.max_model_len == 4096


class TestInferenceConfig:
    """Test InferenceConfig class."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = InferenceConfig()
        
        assert config.max_new_tokens == 2048
        assert config.temperature == 0.7
        assert config.top_p == 0.8
        assert config.top_k == 20
        assert config.repetition_penalty == 1.05
        assert config.do_sample is True
        assert config.stream is True
        assert config.enable_thinking is None
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = InferenceConfig(
            max_new_tokens=1024,
            temperature=0.5,
            enable_thinking=True
        )
        
        assert config.max_new_tokens == 1024
        assert config.temperature == 0.5
        assert config.enable_thinking is True


class TestTrainingConfig:
    """Test TrainingConfig class."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = TrainingConfig(output_dir="test_output")
        
        assert config.output_dir == "test_output"
        assert config.learning_rate == 5e-5
        assert config.num_train_epochs == 3
        assert config.use_lora is False
        assert config.use_qlora is False
    
    def test_lora_config(self):
        """Test LoRA configuration."""
        config = TrainingConfig(
            output_dir="test_output",
            use_lora=True,
            lora_rank=32
        )
        
        assert config.use_lora is True
        assert config.lora_rank == 32
        assert config.lora_alpha == 32
        assert config.lora_target_modules == ["q_proj", "v_proj"]


class TestDeploymentConfig:
    """Test DeploymentConfig class."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = DeploymentConfig()
        
        assert config.framework == "vllm"
        assert config.host == "0.0.0.0"
        assert config.port == 8000
        assert config.num_gpus == 1
        assert config.use_docker is False
    
    def test_docker_config(self):
        """Test Docker configuration."""
        config = DeploymentConfig(
            use_docker=True,
            docker_image="custom:latest"
        )
        
        assert config.use_docker is True
        assert config.docker_image == "custom:latest"


class TestConfigManager:
    """Test ConfigManager class."""
    
    def test_empty_manager(self):
        """Test empty configuration manager."""
        manager = ConfigManager()
        
        model_config = manager.get_model_config(model_name_or_path="test")
        assert model_config.model_name_or_path == "test"
        
        inference_config = manager.get_inference_config()
        assert inference_config.max_new_tokens == 2048
    
    def test_load_from_file(self):
        """Test loading configuration from file."""
        config_data = {
            'model': {
                'model_name_or_path': 'test_model',
                'torch_dtype': 'float16'
            },
            'inference': {
                'max_new_tokens': 1024,
                'temperature': 0.5
            },
            'deployment': {
                'framework': 'sglang',
                'port': 9000
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            config_file = f.name
        
        try:
            manager = ConfigManager(config_file)
            
            model_config = manager.get_model_config()
            assert model_config.model_name_or_path == 'test_model'
            assert model_config.torch_dtype == 'float16'
            
            inference_config = manager.get_inference_config()
            assert inference_config.max_new_tokens == 1024
            assert inference_config.temperature == 0.5
            
            deployment_config = manager.get_deployment_config()
            assert deployment_config.framework == 'sglang'
            assert deployment_config.port == 9000
            
        finally:
            os.unlink(config_file)
    
    def test_save_to_file(self):
        """Test saving configuration to file."""
        manager = ConfigManager()
        manager.update_config('model', model_name_or_path='saved_model')
        manager.update_config('inference', temperature=0.3)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config_file = f.name
        
        try:
            manager.save_to_file(config_file)
            
            # Load and verify
            new_manager = ConfigManager(config_file)
            model_config = new_manager.get_model_config()
            inference_config = new_manager.get_inference_config()
            
            assert model_config.model_name_or_path == 'saved_model'
            assert inference_config.temperature == 0.3
            
        finally:
            os.unlink(config_file)
    
    def test_from_env(self):
        """Test creating manager from environment."""
        # Set environment variable
        config_data = {'model': {'model_name_or_path': 'env_model'}}
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            config_file = f.name
        
        try:
            os.environ['QWEN3_CONFIG_PATH'] = config_file
            manager = ConfigManager.from_env()
            
            model_config = manager.get_model_config()
            assert model_config.model_name_or_path == 'env_model'
            
        finally:
            os.unlink(config_file)
            if 'QWEN3_CONFIG_PATH' in os.environ:
                del os.environ['QWEN3_CONFIG_PATH']
    
    def test_update_config(self):
        """Test updating configuration."""
        manager = ConfigManager()
        
        manager.update_config('inference', max_new_tokens=512)
        inference_config = manager.get_inference_config()
        assert inference_config.max_new_tokens == 512
        
        # Override with kwargs
        inference_config = manager.get_inference_config(max_new_tokens=256)
        assert inference_config.max_new_tokens == 256


if __name__ == "__main__":
    pytest.main([__file__])
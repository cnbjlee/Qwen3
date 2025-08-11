"""
Qwen3: A unified Python library for Qwen3 model inference, training, and deployment.

This package provides:
- Model loading and inference
- Training utilities
- Deployment tools
- Evaluation frameworks
"""

__version__ = "3.2507.0"

from .core import (
    QwenModel,
    QwenTokenizer,
    load_model,
    load_tokenizer,
)

from .inference import (
    InferenceEngine,
    StreamingInference,
    BatchInference,
)

from .training import (
    TrainingConfig,
    FineTuner,
    LoRATrainer,
    QLoRATrainer,
)

from .deployment import (
    DeploymentConfig,
    DockerDeployer,
    VLLMDeployer,
    SGLangDeployer,
)

from .evaluation import (
    EvaluationSuite,
    BenchmarkRunner,
    MetricsCalculator,
)

__all__ = [
    # Core
    "QwenModel",
    "QwenTokenizer", 
    "load_model",
    "load_tokenizer",
    
    # Inference
    "InferenceEngine",
    "StreamingInference",
    "BatchInference",
    
    # Training
    "TrainingConfig",
    "FineTuner",
    "LoRATrainer", 
    "QLoRATrainer",
    
    # Deployment
    "DeploymentConfig",
    "DockerDeployer",
    "VLLMDeployer",
    "SGLangDeployer",
    
    # Evaluation
    "EvaluationSuite",
    "BenchmarkRunner",
    "MetricsCalculator",
]
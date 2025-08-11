"""
Unified inference interface for Qwen3 models.

Supports multiple backends:
- Transformers (HuggingFace)
- vLLM
- SGLang  
- TensorRT-LLM
- llama.cpp
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union, AsyncGenerator, Generator
from dataclasses import asdict

from .config import ModelConfig, InferenceConfig


logger = logging.getLogger(__name__)


class BaseInferenceEngine(ABC):
    """Base class for inference engines."""
    
    def __init__(self, model_config: ModelConfig, inference_config: InferenceConfig):
        self.model_config = model_config
        self.inference_config = inference_config
        self.model = None
        self.tokenizer = None
    
    @abstractmethod
    def load_model(self) -> None:
        """Load model and tokenizer."""
        pass
    
    @abstractmethod
    def generate(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Generate response for given messages."""
        pass
    
    @abstractmethod
    def stream_generate(self, messages: List[Dict[str, str]], **kwargs) -> Generator[str, None, None]:
        """Generate streaming response for given messages."""
        pass
    
    def format_messages(self, messages: List[Dict[str, str]]) -> str:
        """Format messages using chat template."""
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not loaded")
        
        # Handle thinking mode
        enable_thinking = kwargs.get('enable_thinking', self.inference_config.enable_thinking)
        
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking
        )


class TransformersInferenceEngine(BaseInferenceEngine):
    """Transformers-based inference engine."""
    
    def load_model(self) -> None:
        """Load model using Transformers."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            raise ImportError("transformers is required for TransformersInferenceEngine")
        
        logger.info(f"Loading model: {self.model_config.model_name_or_path}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_config.model_name_or_path,
            trust_remote_code=self.model_config.trust_remote_code
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_config.model_name_or_path,
            torch_dtype=self.model_config.torch_dtype,
            device_map=self.model_config.device_map,
            trust_remote_code=self.model_config.trust_remote_code
        ).eval()
        
        # Configure generation
        self.model.generation_config.max_new_tokens = self.inference_config.max_new_tokens
        
        logger.info("Model loaded successfully")
    
    def generate(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Generate response using Transformers."""
        if self.model is None or self.tokenizer is None:
            self.load_model()
        
        # Merge configs
        gen_config = asdict(self.inference_config)
        gen_config.update(kwargs)
        
        input_text = self.format_messages(messages, **kwargs)
        inputs = self.tokenizer([input_text], return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=gen_config['max_new_tokens'],
                temperature=gen_config['temperature'],
                top_p=gen_config['top_p'],
                top_k=gen_config['top_k'],
                repetition_penalty=gen_config['repetition_penalty'],
                do_sample=gen_config['do_sample']
            )
        
        output_ids = generated_ids[0][len(inputs.input_ids[0]):].tolist()
        return self.tokenizer.decode(output_ids, skip_special_tokens=True)
    
    def stream_generate(self, messages: List[Dict[str, str]], **kwargs) -> Generator[str, None, None]:
        """Generate streaming response using Transformers."""
        from transformers import TextIteratorStreamer
        from threading import Thread
        
        if self.model is None or self.tokenizer is None:
            self.load_model()
        
        # Merge configs
        gen_config = asdict(self.inference_config)
        gen_config.update(kwargs)
        
        input_text = self.format_messages(messages, **kwargs)
        inputs = self.tokenizer([input_text], return_tensors="pt").to(self.model.device)
        
        streamer = TextIteratorStreamer(
            tokenizer=self.tokenizer,
            skip_prompt=True,
            timeout=60.0,
            skip_special_tokens=True
        )
        
        generation_kwargs = {
            **inputs,
            "streamer": streamer,
            "max_new_tokens": gen_config['max_new_tokens'],
            "temperature": gen_config['temperature'],
            "top_p": gen_config['top_p'],
            "top_k": gen_config['top_k'],
            "repetition_penalty": gen_config['repetition_penalty'],
            "do_sample": gen_config['do_sample']
        }
        
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()
        
        for new_text in streamer:
            yield new_text


class VLLMInferenceEngine(BaseInferenceEngine):
    """vLLM-based inference engine."""
    
    def load_model(self) -> None:
        """Load model using vLLM."""
        try:
            from vllm import LLM
        except ImportError:
            raise ImportError("vllm is required for VLLMInferenceEngine")
        
        logger.info(f"Loading model with vLLM: {self.model_config.model_name_or_path}")
        
        self.model = LLM(
            model=self.model_config.model_name_or_path,
            trust_remote_code=self.model_config.trust_remote_code,
            max_model_len=self.model_config.max_model_len or 32768,
            gpu_memory_utilization=0.9,
            dtype=self.model_config.torch_dtype
        )
        
        self.tokenizer = self.model.get_tokenizer()
        logger.info("vLLM model loaded successfully")
    
    def generate(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Generate response using vLLM."""
        from vllm import SamplingParams
        
        if self.model is None:
            self.load_model()
        
        # Merge configs
        gen_config = asdict(self.inference_config)
        gen_config.update(kwargs)
        
        input_text = self.format_messages(messages, **kwargs)
        
        sampling_params = SamplingParams(
            max_tokens=gen_config['max_new_tokens'],
            temperature=gen_config['temperature'],
            top_p=gen_config['top_p'],
            top_k=gen_config['top_k'],
            repetition_penalty=gen_config['repetition_penalty']
        )
        
        outputs = self.model.generate([input_text], sampling_params)
        return outputs[0].outputs[0].text
    
    def stream_generate(self, messages: List[Dict[str, str]], **kwargs) -> Generator[str, None, None]:
        """Generate streaming response using vLLM."""
        # vLLM streaming implementation would go here
        # This is a simplified version
        response = self.generate(messages, **kwargs)
        for char in response:
            yield char


class InferenceEngineFactory:
    """Factory for creating inference engines."""
    
    ENGINES = {
        'transformers': TransformersInferenceEngine,
        'vllm': VLLMInferenceEngine,
        # 'sglang': SGLangInferenceEngine,
        # 'tensorrt-llm': TensorRTLLMInferenceEngine,
    }
    
    @classmethod
    def create_engine(
        self,
        engine_type: str,
        model_config: ModelConfig,
        inference_config: InferenceConfig
    ) -> BaseInferenceEngine:
        """Create inference engine by type."""
        if engine_type not in self.ENGINES:
            raise ValueError(f"Unsupported engine type: {engine_type}")
        
        engine_class = self.ENGINES[engine_type]
        return engine_class(model_config, inference_config)


class InferenceManager:
    """High-level inference manager."""
    
    def __init__(self, engine_type: str = 'transformers'):
        self.engine_type = engine_type
        self.engine: Optional[BaseInferenceEngine] = None
    
    def initialize(
        self,
        model_name_or_path: str,
        model_config: Optional[ModelConfig] = None,
        inference_config: Optional[InferenceConfig] = None
    ) -> None:
        """Initialize inference engine."""
        if model_config is None:
            model_config = ModelConfig(model_name_or_path=model_name_or_path)
        
        if inference_config is None:
            inference_config = InferenceConfig()
        
        self.engine = InferenceEngineFactory.create_engine(
            self.engine_type,
            model_config,
            inference_config
        )
        
        self.engine.load_model()
    
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Chat with the model."""
        if self.engine is None:
            raise RuntimeError("Engine not initialized")
        
        return self.engine.generate(messages, **kwargs)
    
    def stream_chat(self, messages: List[Dict[str, str]], **kwargs) -> Generator[str, None, None]:
        """Stream chat with the model."""
        if self.engine is None:
            raise RuntimeError("Engine not initialized")
        
        return self.engine.stream_generate(messages, **kwargs)
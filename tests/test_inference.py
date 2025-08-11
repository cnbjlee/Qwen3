"""
Integration tests for inference engines.
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

sys.path.append(str(Path(__file__).parent.parent))

from qwen3.config import ModelConfig, InferenceConfig
from qwen3.inference import (
    TransformersInferenceEngine,
    InferenceEngineFactory,
    InferenceManager
)


class TestTransformersInferenceEngine:
    """Test TransformersInferenceEngine."""
    
    @pytest.fixture
    def mock_model_config(self):
        """Mock model configuration."""
        return ModelConfig(
            model_name_or_path="mock_model",
            device_map="cpu"
        )
    
    @pytest.fixture
    def mock_inference_config(self):
        """Mock inference configuration."""
        return InferenceConfig(
            max_new_tokens=100,
            temperature=0.7
        )
    
    @patch('qwen3.inference.AutoModelForCausalLM')
    @patch('qwen3.inference.AutoTokenizer')
    def test_load_model(self, mock_tokenizer_class, mock_model_class, 
                       mock_model_config, mock_inference_config):
        """Test model loading."""
        # Mock tokenizer and model
        mock_tokenizer = Mock()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        mock_model = Mock()
        mock_model.eval.return_value = mock_model
        mock_model.generation_config = Mock()
        mock_model_class.from_pretrained.return_value = mock_model
        
        # Create engine and load model
        engine = TransformersInferenceEngine(mock_model_config, mock_inference_config)
        engine.load_model()
        
        # Verify calls
        mock_tokenizer_class.from_pretrained.assert_called_once_with(
            "mock_model",
            trust_remote_code=True
        )
        
        mock_model_class.from_pretrained.assert_called_once_with(
            "mock_model",
            torch_dtype="auto",
            device_map="cpu",
            trust_remote_code=True
        )
        
        assert engine.tokenizer == mock_tokenizer
        assert engine.model == mock_model
        assert mock_model.generation_config.max_new_tokens == 100
    
    @patch('qwen3.inference.AutoModelForCausalLM')
    @patch('qwen3.inference.AutoTokenizer')
    @patch('torch.no_grad')
    def test_generate(self, mock_no_grad, mock_tokenizer_class, mock_model_class,
                     mock_model_config, mock_inference_config):
        """Test text generation."""
        # Mock tokenizer
        mock_tokenizer = Mock()
        mock_tokenizer.apply_chat_template.return_value = "formatted_input"
        mock_tokenizer.return_value = {"input_ids": Mock(), "attention_mask": Mock()}
        mock_tokenizer.decode.return_value = "Generated response"
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        # Mock model
        mock_model = Mock()
        mock_model.eval.return_value = mock_model
        mock_model.generation_config = Mock()
        mock_model.device = "cpu"
        mock_model.generate.return_value = [Mock()]
        mock_model_class.from_pretrained.return_value = mock_model
        
        # Mock inputs
        mock_inputs = Mock()
        mock_inputs.input_ids = [Mock()]
        mock_tokenizer.return_value = mock_inputs
        
        # Create engine
        engine = TransformersInferenceEngine(mock_model_config, mock_inference_config)
        engine.load_model()
        
        # Test generation
        messages = [{"role": "user", "content": "Hello"}]
        with patch.object(engine, 'format_messages', return_value="formatted"):
            response = engine.generate(messages)
        
        assert response == "Generated response"
        mock_model.generate.assert_called_once()
    
    @patch('qwen3.inference.TextIteratorStreamer')
    @patch('qwen3.inference.Thread')
    @patch('qwen3.inference.AutoModelForCausalLM')
    @patch('qwen3.inference.AutoTokenizer')
    def test_stream_generate(self, mock_tokenizer_class, mock_model_class,
                           mock_thread_class, mock_streamer_class,
                           mock_model_config, mock_inference_config):
        """Test streaming generation."""
        # Mock streamer
        mock_streamer = Mock()
        mock_streamer.__iter__ = Mock(return_value=iter(["chunk1", "chunk2", "chunk3"]))
        mock_streamer_class.return_value = mock_streamer
        
        # Mock thread
        mock_thread = Mock()
        mock_thread_class.return_value = mock_thread
        
        # Mock tokenizer and model
        mock_tokenizer = Mock()
        mock_tokenizer.apply_chat_template.return_value = "formatted_input"
        mock_tokenizer.return_value = Mock()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        mock_model = Mock()
        mock_model.eval.return_value = mock_model
        mock_model.generation_config = Mock()
        mock_model.device = "cpu"
        mock_model_class.from_pretrained.return_value = mock_model
        
        # Create engine
        engine = TransformersInferenceEngine(mock_model_config, mock_inference_config)
        engine.load_model()
        
        # Test streaming
        messages = [{"role": "user", "content": "Hello"}]
        with patch.object(engine, 'format_messages', return_value="formatted"):
            chunks = list(engine.stream_generate(messages))
        
        assert chunks == ["chunk1", "chunk2", "chunk3"]
        mock_thread.start.assert_called_once()


class TestInferenceEngineFactory:
    """Test InferenceEngineFactory."""
    
    def test_create_transformers_engine(self):
        """Test creating Transformers engine."""
        model_config = ModelConfig(model_name_or_path="test")
        inference_config = InferenceConfig()
        
        engine = InferenceEngineFactory.create_engine(
            "transformers", model_config, inference_config
        )
        
        assert isinstance(engine, TransformersInferenceEngine)
        assert engine.model_config == model_config
        assert engine.inference_config == inference_config
    
    def test_unsupported_engine(self):
        """Test unsupported engine type."""
        model_config = ModelConfig(model_name_or_path="test")
        inference_config = InferenceConfig()
        
        with pytest.raises(ValueError, match="Unsupported engine type"):
            InferenceEngineFactory.create_engine(
                "unsupported", model_config, inference_config
            )


class TestInferenceManager:
    """Test InferenceManager."""
    
    @patch('qwen3.inference.InferenceEngineFactory.create_engine')
    def test_initialize(self, mock_create_engine):
        """Test manager initialization."""
        mock_engine = Mock()
        mock_create_engine.return_value = mock_engine
        
        manager = InferenceManager("transformers")
        manager.initialize("test_model")
        
        assert manager.engine == mock_engine
        mock_engine.load_model.assert_called_once()
        mock_create_engine.assert_called_once()
    
    @patch('qwen3.inference.InferenceEngineFactory.create_engine')
    def test_chat(self, mock_create_engine):
        """Test chat functionality."""
        mock_engine = Mock()
        mock_engine.generate.return_value = "Response"
        mock_create_engine.return_value = mock_engine
        
        manager = InferenceManager("transformers")
        manager.initialize("test_model")
        
        messages = [{"role": "user", "content": "Hello"}]
        response = manager.chat(messages)
        
        assert response == "Response"
        mock_engine.generate.assert_called_once_with(messages)
    
    @patch('qwen3.inference.InferenceEngineFactory.create_engine')
    def test_stream_chat(self, mock_create_engine):
        """Test streaming chat functionality."""
        mock_engine = Mock()
        mock_engine.stream_generate.return_value = iter(["chunk1", "chunk2"])
        mock_create_engine.return_value = mock_engine
        
        manager = InferenceManager("transformers")
        manager.initialize("test_model")
        
        messages = [{"role": "user", "content": "Hello"}]
        chunks = list(manager.stream_chat(messages))
        
        assert chunks == ["chunk1", "chunk2"]
        mock_engine.stream_generate.assert_called_once_with(messages)
    
    def test_chat_without_initialization(self):
        """Test chat without initialization."""
        manager = InferenceManager("transformers")
        
        with pytest.raises(RuntimeError, match="Engine not initialized"):
            manager.chat([])
        
        with pytest.raises(RuntimeError, match="Engine not initialized"):
            list(manager.stream_chat([]))


if __name__ == "__main__":
    pytest.main([__file__])
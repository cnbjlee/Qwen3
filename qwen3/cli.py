"""
Command-line interface for Qwen3.

Provides unified commands for:
- Model inference
- Model deployment
- Model evaluation
- Model training
"""

import argparse
import sys
import json
from pathlib import Path
from typing import List, Dict, Any

from .config import ConfigManager, ModelConfig, InferenceConfig, DeploymentConfig
from .inference import InferenceManager
from .deployment import DeploymentManager
from .monitoring import QwenLogger


def cmd_chat(args) -> None:
    """Interactive chat command."""
    from .inference import InferenceManager
    
    # Setup inference manager
    model_config = ModelConfig(
        model_name_or_path=args.model,
        device_map="cpu" if args.cpu_only else "auto"
    )
    
    inference_config = InferenceConfig(
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        enable_thinking=args.thinking
    )
    
    manager = InferenceManager(engine_type=args.backend)
    
    print("Loading model...")
    try:
        manager.initialize(
            model_name_or_path=args.model,
            model_config=model_config,
            inference_config=inference_config
        )
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    
    print("Type 'exit' to quit, 'help' for commands")
    
    while True:
        try:
            user_input = input("\n👤 You: ").strip()
            
            if user_input.lower() in ['exit', 'quit']:
                break
            elif user_input.lower() == 'help':
                print("Commands:")
                print("  exit/quit - Exit the chat")
                print("  help      - Show this help")
                continue
            elif not user_input:
                continue
            
            messages = [{"role": "user", "content": user_input}]
            
            print("🤖 Qwen: ", end="", flush=True)
            if args.stream:
                response = ""
                for chunk in manager.stream_chat(messages):
                    print(chunk, end="", flush=True)
                    response += chunk
                print()
            else:
                response = manager.chat(messages)
                print(response)
                
        except KeyboardInterrupt:
            print("\n\nGoodbye! 👋")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")


def cmd_deploy(args) -> None:
    """Deploy model command."""
    config = DeploymentConfig(
        framework=args.framework,
        model_name_or_path=args.model,
        host=args.host,
        port=args.port,
        num_gpus=args.num_gpus,
        max_model_len=args.max_model_len,
        use_docker=args.docker
    )
    
    manager = DeploymentManager(config)
    
    if args.action == "start":
        print(f"Deploying {args.model} with {args.framework}...")
        success = manager.deploy()
        if success:
            print("🚀 Deployment successful!")
            print(f"📡 Endpoint: http://{args.host}:{args.port}")
        else:
            print("❌ Deployment failed!")
            sys.exit(1)
            
    elif args.action == "stop":
        manager.stop()
        print("🛑 Service stopped")
        
    elif args.action == "status":
        status = manager.status()
        print(f"Status: {json.dumps(status, indent=2)}")


def cmd_eval(args) -> None:
    """Evaluate model command."""
    print("🧪 Starting evaluation...")
    
    # TODO: Implement evaluation logic
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Batch size: {args.batch_size}")
    
    print("⚠️  Evaluation functionality is under development")


def cmd_train(args) -> None:
    """Train model command."""
    print("🎯 Starting training...")
    
    # TODO: Implement training logic
    print(f"Model: {args.base_model}")
    print(f"Dataset: {args.dataset}")
    print(f"Output: {args.output_dir}")
    print(f"Method: {args.method}")
    
    print("⚠️  Training functionality is under development")


def cmd_convert(args) -> None:
    """Convert model format command."""
    print("🔄 Converting model format...")
    
    # TODO: Implement conversion logic
    print(f"Input: {args.input_path}")
    print(f"Output: {args.output_path}")
    print(f"Format: {args.format}")
    
    print("⚠️  Conversion functionality is under development")


def create_parser() -> argparse.ArgumentParser:
    """Create command-line parser."""
    parser = argparse.ArgumentParser(
        prog="qwen3",
        description="Qwen3 - A unified toolkit for large language models"
    )
    
    parser.add_argument(
        "--version", 
        action="version", 
        version="%(prog)s 3.2507.0"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Chat command
    chat_parser = subparsers.add_parser("chat", help="Interactive chat with model")
    chat_parser.add_argument("--model", "-m", default="Qwen/Qwen3-8B-Instruct", help="Model name or path")
    chat_parser.add_argument("--backend", choices=["transformers", "vllm", "sglang"], default="transformers", help="Inference backend")
    chat_parser.add_argument("--max-tokens", type=int, default=2048, help="Maximum tokens to generate")
    chat_parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    chat_parser.add_argument("--thinking", action="store_true", help="Enable thinking mode")
    chat_parser.add_argument("--no-stream", dest="stream", action="store_false", help="Disable streaming")
    chat_parser.add_argument("--cpu-only", action="store_true", help="Use CPU only")
    chat_parser.set_defaults(func=cmd_chat)
    
    # Deploy command
    deploy_parser = subparsers.add_parser("deploy", help="Deploy model as service")
    deploy_parser.add_argument("action", choices=["start", "stop", "status"], help="Deployment action")
    deploy_parser.add_argument("--model", "-m", default="Qwen/Qwen3-8B-Instruct", help="Model name or path")
    deploy_parser.add_argument("--framework", choices=["vllm", "sglang", "tensorrt", "docker"], default="vllm", help="Deployment framework")
    deploy_parser.add_argument("--host", default="0.0.0.0", help="Host address")
    deploy_parser.add_argument("--port", type=int, default=8000, help="Service port")
    deploy_parser.add_argument("--num-gpus", type=int, default=1, help="Number of GPUs")
    deploy_parser.add_argument("--max-model-len", type=int, default=32768, help="Maximum model length")
    deploy_parser.add_argument("--docker", action="store_true", help="Use Docker deployment")
    deploy_parser.set_defaults(func=cmd_deploy)
    
    # Eval command
    eval_parser = subparsers.add_parser("eval", help="Evaluate model performance")
    eval_parser.add_argument("--model", "-m", required=True, help="Model name or path")
    eval_parser.add_argument("--dataset", "-d", required=True, help="Evaluation dataset")
    eval_parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    eval_parser.add_argument("--output", "-o", help="Output file for results")
    eval_parser.set_defaults(func=cmd_eval)
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Fine-tune model")
    train_parser.add_argument("--base-model", "-m", required=True, help="Base model name or path")
    train_parser.add_argument("--dataset", "-d", required=True, help="Training dataset")
    train_parser.add_argument("--output-dir", "-o", required=True, help="Output directory")
    train_parser.add_argument("--method", choices=["full", "lora", "qlora"], default="lora", help="Training method")
    train_parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    train_parser.add_argument("--learning-rate", type=float, default=5e-5, help="Learning rate")
    train_parser.set_defaults(func=cmd_train)
    
    # Convert command
    convert_parser = subparsers.add_parser("convert", help="Convert model format")
    convert_parser.add_argument("input_path", help="Input model path")
    convert_parser.add_argument("output_path", help="Output model path")
    convert_parser.add_argument("--format", choices=["gguf", "awq", "gptq"], required=True, help="Target format")
    convert_parser.add_argument("--quantization", choices=["int4", "int8", "fp16"], help="Quantization level")
    convert_parser.set_defaults(func=cmd_convert)
    
    return parser


def main(argv: List[str] = None) -> None:
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args(argv)
    
    if not hasattr(args, 'func'):
        parser.print_help()
        return
    
    # Setup logging
    logger = QwenLogger()
    
    try:
        args.func(args)
    except KeyboardInterrupt:
        print("\n🛑 Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.log_error(e)
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
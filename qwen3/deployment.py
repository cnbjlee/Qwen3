"""
Unified deployment manager for Qwen3 models.

Supports:
- Docker deployment
- vLLM deployment  
- SGLang deployment
- TensorRT-LLM deployment
- Local deployment
"""

import os
import subprocess
import time
import requests
import docker
import yaml
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from pathlib import Path

from .config import DeploymentConfig


class BaseDeployer(ABC):
    """Base class for deployers."""
    
    def __init__(self, config: DeploymentConfig):
        self.config = config
        self.process = None
        self.container = None
    
    @abstractmethod
    def deploy(self) -> None:
        """Deploy the model."""
        pass
    
    @abstractmethod
    def stop(self) -> None:
        """Stop the deployment."""
        pass
    
    @abstractmethod
    def health_check(self) -> bool:
        """Check if the service is healthy."""
        pass
    
    def wait_for_ready(self, timeout: int = 300) -> bool:
        """Wait for the service to be ready."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.health_check():
                return True
            time.sleep(5)
        return False


class VLLMDeployer(BaseDeployer):
    """vLLM deployer."""
    
    def deploy(self) -> None:
        """Deploy using vLLM."""
        cmd = [
            "python", "-m", "vllm.entrypoints.openai.api_server",
            "--model", self.config.model_name_or_path,
            "--host", self.config.host,
            "--port", str(self.config.port),
            "--tensor-parallel-size", str(self.config.num_gpus),
            "--max-model-len", str(self.config.max_model_len),
            "--gpu-memory-utilization", str(self.config.gpu_memory_utilization),
        ]
        
        if self.config.enable_prefix_caching:
            cmd.append("--enable-prefix-caching")
        
        if self.config.disable_log_requests:
            cmd.append("--disable-log-requests")
        
        print(f"Starting vLLM server: {' '.join(cmd)}")
        self.process = subprocess.Popen(cmd)
    
    def stop(self) -> None:
        """Stop vLLM server."""
        if self.process:
            self.process.terminate()
            self.process.wait()
            self.process = None
    
    def health_check(self) -> bool:
        """Check vLLM server health."""
        try:
            response = requests.get(f"http://{self.config.host}:{self.config.port}/health")
            return response.status_code == 200
        except requests.RequestException:
            return False


class SGLangDeployer(BaseDeployer):
    """SGLang deployer."""
    
    def deploy(self) -> None:
        """Deploy using SGLang."""
        cmd = [
            "python", "-m", "sglang.launch_server",
            "--model-path", self.config.model_name_or_path,
            "--host", self.config.host,
            "--port", str(self.config.port),
            "--context-length", str(self.config.max_model_len),
        ]
        
        # Add reasoning parser for thinking models
        if "thinking" in self.config.model_name_or_path.lower():
            cmd.extend(["--reasoning-parser", "deepseek-r1"])
        elif "qwen3" in self.config.model_name_or_path.lower():
            cmd.extend(["--reasoning-parser", "qwen3"])
        
        print(f"Starting SGLang server: {' '.join(cmd)}")
        self.process = subprocess.Popen(cmd)
    
    def stop(self) -> None:
        """Stop SGLang server."""
        if self.process:
            self.process.terminate()
            self.process.wait()
            self.process = None
    
    def health_check(self) -> bool:
        """Check SGLang server health."""
        try:
            response = requests.get(f"http://{self.config.host}:{self.config.port}/health")
            return response.status_code == 200
        except requests.RequestException:
            return False


class DockerDeployer(BaseDeployer):
    """Docker deployer."""
    
    def __init__(self, config: DeploymentConfig):
        super().__init__(config)
        self.client = docker.from_env()
    
    def deploy(self) -> None:
        """Deploy using Docker."""
        # Build image if needed
        if not self._image_exists():
            self._build_image()
        
        # Run container
        environment = {
            "MODEL_NAME": self.config.model_name_or_path,
            "HOST": self.config.host,
            "PORT": str(self.config.port),
            "NUM_GPUS": str(self.config.num_gpus),
        }
        
        self.container = self.client.containers.run(
            self.config.docker_image,
            detach=True,
            ports=self.config.docker_ports,
            environment=environment,
            device_requests=[
                docker.types.DeviceRequest(count=self.config.num_gpus, capabilities=[["gpu"]])
            ] if self.config.num_gpus > 0 else None,
            name=f"qwen3-{int(time.time())}"
        )
        
        print(f"Docker container started: {self.container.id}")
    
    def stop(self) -> None:
        """Stop Docker container."""
        if self.container:
            self.container.stop()
            self.container.remove()
            self.container = None
    
    def health_check(self) -> bool:
        """Check Docker container health."""
        if not self.container:
            return False
        
        try:
            self.container.reload()
            return self.container.status == "running"
        except docker.errors.NotFound:
            return False
    
    def _image_exists(self) -> bool:
        """Check if Docker image exists."""
        try:
            self.client.images.get(self.config.docker_image)
            return True
        except docker.errors.ImageNotFound:
            return False
    
    def _build_image(self) -> None:
        """Build Docker image."""
        dockerfile_path = Path(__file__).parent.parent / "docker" / "Dockerfile-cu121"
        build_context = Path(__file__).parent.parent
        
        print(f"Building Docker image: {self.config.docker_image}")
        self.client.images.build(
            path=str(build_context),
            dockerfile=str(dockerfile_path),
            tag=self.config.docker_image,
            rm=True
        )


class DeploymentManager:
    """Unified deployment manager."""
    
    DEPLOYERS = {
        'vllm': VLLMDeployer,
        'sglang': SGLangDeployer,
        'docker': DockerDeployer,
    }
    
    def __init__(self, config: DeploymentConfig):
        self.config = config
        self.deployer: Optional[BaseDeployer] = None
    
    def deploy(self) -> bool:
        """Deploy the model."""
        if self.config.framework not in self.DEPLOYERS:
            raise ValueError(f"Unsupported framework: {self.config.framework}")
        
        deployer_class = self.DEPLOYERS[self.config.framework]
        self.deployer = deployer_class(self.config)
        
        try:
            self.deployer.deploy()
            
            print("Waiting for service to be ready...")
            if self.deployer.wait_for_ready():
                print("✅ Service is ready!")
                return True
            else:
                print("❌ Service failed to start within timeout")
                return False
                
        except Exception as e:
            print(f"❌ Deployment failed: {e}")
            return False
    
    def stop(self) -> None:
        """Stop the deployment."""
        if self.deployer:
            self.deployer.stop()
            print("🛑 Service stopped")
    
    def status(self) -> Dict[str, Any]:
        """Get deployment status."""
        if not self.deployer:
            return {"status": "not_deployed"}
        
        return {
            "status": "running" if self.deployer.health_check() else "stopped",
            "framework": self.config.framework,
            "endpoint": f"http://{self.config.host}:{self.config.port}",
            "model": self.config.model_name_or_path,
        }
    
    @classmethod
    def from_config_file(cls, config_path: str) -> 'DeploymentManager':
        """Create deployment manager from config file."""
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        config = DeploymentConfig(**config_dict.get('deployment', {}))
        return cls(config)


# CLI interface
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Qwen3 Deployment Manager")
    parser.add_argument("action", choices=["deploy", "stop", "status"], help="Action to perform")
    parser.add_argument("--config", help="Configuration file path")
    parser.add_argument("--model", help="Model name or path")
    parser.add_argument("--framework", choices=["vllm", "sglang", "docker"], default="vllm", help="Deployment framework")
    parser.add_argument("--port", type=int, default=8000, help="Service port")
    parser.add_argument("--num-gpus", type=int, default=1, help="Number of GPUs")
    
    args = parser.parse_args()
    
    if args.config:
        manager = DeploymentManager.from_config_file(args.config)
    else:
        config = DeploymentConfig(
            framework=args.framework,
            model_name_or_path=args.model or "Qwen/Qwen3-8B-Instruct",
            port=args.port,
            num_gpus=args.num_gpus
        )
        manager = DeploymentManager(config)
    
    if args.action == "deploy":
        manager.deploy()
    elif args.action == "stop":
        manager.stop()
    elif args.action == "status":
        status = manager.status()
        print(f"Status: {status}")


if __name__ == "__main__":
    main()
# Copyright (c) Alibaba Cloud.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""A simple command-line interactive chat demo."""

import argparse
import os
import platform
import shutil
from copy import deepcopy
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from qwen3.inference import InferenceManager
from qwen3.config import ModelConfig, InferenceConfig
from transformers.trainer_utils import set_seed

DEFAULT_CKPT_PATH = "Qwen/Qwen3-8B-Instruct"

_WELCOME_MSG = """\
Welcome to use Qwen2.5-Instruct model, type text to start chat, type :h to show command help.
(欢迎使用 Qwen2.5-Instruct 模型，输入内容即可进行对话，:h 显示命令帮助。)

Note: This demo is governed by the original license of Qwen2.5.
We strongly advise users not to knowingly generate or allow others to knowingly generate harmful content, including hate speech, violence, pornography, deception, etc.
(注：本演示受Qwen2.5的许可协议限制。我们强烈建议，用户不应传播及不应允许他人传播以下内容，包括但不限于仇恨言论、暴力、色情、欺诈相关的有害信息。)
"""
_HELP_MSG = """\
Commands:
    :help / :h              Show this help message              显示帮助信息
    :exit / :quit / :q      Exit the demo                       退出Demo
    :clear / :cl            Clear screen                        清屏
    :clear-history / :clh   Clear history                       清除对话历史
    :history / :his         Show history                        显示对话历史
    :seed                   Show current random seed            显示当前随机种子
    :seed <N>               Set random seed to <N>              设置随机种子
    :conf                   Show current generation config      显示生成配置
    :conf <key>=<value>     Change generation config            修改生成配置
    :reset-conf             Reset generation config             重置生成配置
"""
_ALL_COMMAND_NAMES = [
    "help",
    "h",
    "exit",
    "quit",
    "q",
    "clear",
    "cl",
    "clear-history",
    "clh",
    "history",
    "his",
    "seed",
    "conf",
    "reset-conf",
]


def _setup_readline():
    try:
        import readline
    except ImportError:
        return

    _matches = []

    def _completer(text, state):
        nonlocal _matches

        if state == 0:
            _matches = [
                cmd_name for cmd_name in _ALL_COMMAND_NAMES if cmd_name.startswith(text)
            ]
        if 0 <= state < len(_matches):
            return _matches[state]
        return None

    readline.set_completer(_completer)
    readline.parse_and_bind("tab: complete")


def _setup_inference_manager(args):
    """Setup inference manager with configurations."""
    model_config = ModelConfig(
        model_name_or_path=args.checkpoint_path,
        device_map="cpu" if args.cpu_only else "auto"
    )
    
    inference_config = InferenceConfig(
        max_new_tokens=2048,
        enable_thinking=getattr(args, 'enable_thinking', None)
    )
    
    manager = InferenceManager(engine_type='transformers')
    manager.initialize(
        model_name_or_path=args.checkpoint_path,
        model_config=model_config,
        inference_config=inference_config
    )
    
    return manager


def _gc():
    import gc
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _clear_screen():
    if platform.system() == "Windows":
        os.system("cls")
    else:
        os.system("clear")


def _print_history(history):
    terminal_width = shutil.get_terminal_size()[0]
    print(f"History ({len(history)})".center(terminal_width, "="))
    for index, (query, response) in enumerate(history):
        print(f"User[{index}]: {query}")
        print(f"Qwen[{index}]: {response}")
    print("=" * terminal_width)


def _get_input() -> str:
    while True:
        try:
            message = input("User> ").strip()
        except UnicodeDecodeError:
            print("[ERROR] Encoding error in input")
            continue
        except KeyboardInterrupt:
            exit(1)
        if message:
            return message
        print("[ERROR] Query is empty")


def _chat_stream(inference_manager, query, history):
    """Generate streaming chat response."""
    conversation = []
    for query_h, response_h in history:
        conversation.append({"role": "user", "content": query_h})
        conversation.append({"role": "assistant", "content": response_h})
    conversation.append({"role": "user", "content": query})
    
    for new_text in inference_manager.stream_chat(conversation):
        yield new_text


def main():
    parser = argparse.ArgumentParser(
        description="Qwen3-Instruct command-line interactive chat demo."
    )
    parser.add_argument(
        "-c",
        "--checkpoint-path",
        type=str,
        default=DEFAULT_CKPT_PATH,
        help="Checkpoint name or path, default to %(default)r",
    )
    parser.add_argument("-s", "--seed", type=int, default=1234, help="Random seed")
    parser.add_argument(
        "--cpu-only", action="store_true", help="Run demo with CPU only"
    )
    parser.add_argument(
        "--enable-thinking", action="store_true", help="Enable thinking mode"
    )
    parser.add_argument(
        "--disable-thinking", action="store_true", help="Disable thinking mode"
    )
    args = parser.parse_args()
    
    # Handle thinking mode
    if args.enable_thinking:
        args.enable_thinking = True
    elif args.disable_thinking:
        args.enable_thinking = False
    else:
        args.enable_thinking = None  # Use model default

    history, response = [], ""

    try:
        inference_manager = _setup_inference_manager(args)
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return

    _setup_readline()

    _clear_screen()
    print(_WELCOME_MSG)

    seed = args.seed

    while True:
        query = _get_input()

        # Process commands.
        if query.startswith(":"):
            command_words = query[1:].strip().split()
            if not command_words:
                command = ""
            else:
                command = command_words[0]

            if command in ["exit", "quit", "q"]:
                break
            elif command in ["clear", "cl"]:
                _clear_screen()
                print(_WELCOME_MSG)
                _gc()
                continue
            elif command in ["clear-history", "clh"]:
                print(f"[INFO] All {len(history)} history cleared")
                history.clear()
                _gc()
                continue
            elif command in ["help", "h"]:
                print(_HELP_MSG)
                continue
            elif command in ["history", "his"]:
                _print_history(history)
                continue
            elif command in ["seed"]:
                if len(command_words) == 1:
                    print(f"[INFO] Current random seed: {seed}")
                    continue
                else:
                    new_seed_s = command_words[1]
                    try:
                        new_seed = int(new_seed_s)
                    except ValueError:
                        print(
                            f"[WARNING] Fail to change random seed: {new_seed_s!r} is not a valid number"
                        )
                    else:
                        print(f"[INFO] Random seed changed to {new_seed}")
                        seed = new_seed
                    continue
            elif command in ["conf"]:
                if len(command_words) == 1:
                    # Show current config
                    config = inference_manager.engine.inference_config
                    print(f"max_new_tokens: {config.max_new_tokens}")
                    print(f"temperature: {config.temperature}")
                    print(f"top_p: {config.top_p}")
                    print(f"top_k: {config.top_k}")
                    print(f"repetition_penalty: {config.repetition_penalty}")
                    print(f"do_sample: {config.do_sample}")
                else:
                    for key_value_pairs_str in command_words[1:]:
                        eq_idx = key_value_pairs_str.find("=")
                        if eq_idx == -1:
                            print("[WARNING] format: <key>=<value>")
                            continue
                        conf_key, conf_value_str = (
                            key_value_pairs_str[:eq_idx],
                            key_value_pairs_str[eq_idx + 1 :],
                        )
                        try:
                            conf_value = eval(conf_value_str)
                        except Exception as e:
                            print(e)
                            continue
                        else:
                            print(
                                f"[INFO] Change config: {conf_key} = {conf_value}"
                            )
                            setattr(inference_manager.engine.inference_config, conf_key, conf_value)
                continue
            elif command in ["reset-conf"]:
                print("[INFO] Reset generation config")
                inference_manager.engine.inference_config = InferenceConfig()
                print("Configuration reset to default")
                continue
            else:
                # As normal query.
                pass

        # Run chat.
        set_seed(seed)
        _clear_screen()
        print(f"\nUser: {query}")
        print(f"\nQwen: ", end="")
        try:
            partial_text = ""
            for new_text in _chat_stream(inference_manager, query, history):
                print(new_text, end="", flush=True)
                partial_text += new_text
            response = partial_text
            print()

        except KeyboardInterrupt:
            print("[WARNING] Generation interrupted")
            continue

        history.append((query, response))


if __name__ == "__main__":
    main()

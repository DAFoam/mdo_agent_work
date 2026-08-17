#!/usr/bin/env python3
"""Create a customized Ollama model from an installed base model.

Requirements
------------
Install Ollama for Windows, macOS, or Linux, start the Ollama service if needed,
and ensure the ``ollama`` command is available in your terminal. The source model
must already exist locally; for example, run ``ollama pull qwen3.5:4b`` first.

Usage
-----
Run this file from a terminal with Python 3::

    python .create_custom_model.py NAME_ORIG_LLM NAME_NEW_LLM [OPTIONS]

``NAME_ORIG_LLM`` is the installed base model and ``NAME_NEW_LLM`` is the name
for the newly created model. For example::

    python .create_custom_model.py qwen3.5:4b my-qwen
    python .create_custom_model.py qwen3.5:4b my-qwen --num_ctx 65536 --temperature 0
    python .create_custom_model.py gemma4:e2b gemma-creative --temperature 0.8 --top_p 0.9

Every option is optional. An option that is not supplied is deliberately omitted
from the generated Modelfile, so its setting remains inherited from the base
model. This script supports all parameters found in this project's ModelFiles:
``num_ctx``, ``temperature``, ``top_p``, ``top_k``, ``min_p``,
``presence_penalty``, and ``repeat_penalty``. Use ``--help`` to display the
complete command-line reference.
"""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path


PARAMETERS = (
    "num_ctx",
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "presence_penalty",
    "repeat_penalty",
)


def parse_arguments() -> argparse.Namespace:
    """Parse the source model, destination model, and optional Ollama parameters.

    Returns:
        Parsed command-line values. Parameters omitted from the command are ``None``.
    """
    parser = argparse.ArgumentParser(
        description="Create an Ollama model while preserving omitted source-model parameters.",
        epilog=(
            "Example: python .create_custom_model.py qwen3.5:4b my-qwen "
            "--num_ctx 65536 --temperature 0\n\n"
            "Only supplied options are written to the Modelfile. All other settings "
            "remain inherited from the source model."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("name_orig_llm", help="Existing Ollama model, for example qwen3.5:4b")
    parser.add_argument("name_new_llm", help="Name for the customized Ollama model")
    parser.add_argument(
        "--num_ctx",
        dest="num_ctx",
        type=int,
        help="Context-window size (Ollama PARAMETER num_ctx)",
    )
    parser.add_argument("--temperature", type=float, help="Sampling temperature")
    parser.add_argument("--top_p", dest="top_p", type=float, help="Top-p sampling value")
    parser.add_argument("--top_k", dest="top_k", type=int, help="Top-k sampling value")
    parser.add_argument("--min_p", dest="min_p", type=float, help="Minimum-p sampling value")
    parser.add_argument(
        "--presence_penalty", dest="presence_penalty", type=float, help="Presence penalty"
    )
    parser.add_argument(
        "--repeat_penalty", dest="repeat_penalty", type=float, help="Repeat penalty"
    )
    return parser.parse_args()


def build_modelfile(source_model: str, arguments: argparse.Namespace) -> str:
    """Build a Modelfile containing only requested changes.

    Args:
        source_model: Existing Ollama model to customize.
        arguments: Parsed command-line values.

    Returns:
        Text for an Ollama Modelfile.
    """
    lines = [f"FROM {source_model}"]
    # Omitted CLI options intentionally add no line, preserving the source model's setting.
    for parameter in PARAMETERS:
        value = getattr(arguments, parameter)
        if value is not None:
            lines.append(f"PARAMETER {parameter} {value}")
    return "\n".join(lines) + "\n"


def create_model(model_name: str, modelfile_contents: str) -> None:
    """Pass a temporary Modelfile to ``ollama create`` and remove it afterwards.

    Args:
        model_name: Destination Ollama model name.
        modelfile_contents: Complete generated Modelfile text.
    """
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".Modelfile", delete=False) as file:
        file.write(modelfile_contents)
        modelfile_path = Path(file.name)

    try:
        subprocess.run(["ollama", "create", model_name, "-f", str(modelfile_path)], check=True)
    except FileNotFoundError:
        sys.exit("Ollama was not found. Install Ollama and ensure the 'ollama' command is on PATH.")
    except subprocess.CalledProcessError as error:
        sys.exit(error.returncode)
    finally:
        modelfile_path.unlink(missing_ok=True)


def main() -> None:
    """Generate the requested Modelfile and create the customized Ollama model."""
    arguments = parse_arguments()
    create_model(arguments.name_new_llm, build_modelfile(arguments.name_orig_llm, arguments))


if __name__ == "__main__":
    main()

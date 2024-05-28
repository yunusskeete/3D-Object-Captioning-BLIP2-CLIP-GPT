import os
import subprocess
from typing import List

import yaml

ENCODING = "utf-8"


def load_config(config_path):
    with open(config_path, "r", encoding=ENCODING) as file:
        return yaml.safe_load(file)


def find_boolean_value(input_value):
    if isinstance(input_value, bool):
        return input_value

    if isinstance(input_value, (int, float)):
        if input_value == 1:
            return True
        elif input_value == 0:
            return False

    if isinstance(input_value, str):
        if input_value.lower() == "true":
            return True
        elif input_value.lower() == "false":
            return False

    return False


def main():
    config = load_config("config.yaml")

    # Arguments for generate_captions.py
    generate_captions_args = config["generate_captions"]
    _generate_captions_command: List[str, None] = [
        "python",
        "generate_captions.py",
        "--abs_path_to_objects",
        generate_captions_args["abs_path_to_objects"],
        "--output_dir",
        generate_captions_args["output_dir"],
        "--model_name",
        generate_captions_args["model_name"],
        "--model_type",
        generate_captions_args["model_type"],
        "--use_qa" if find_boolean_value(generate_captions_args["use_qa"]) else None,
        "--prompt",
        generate_captions_args["prompt"],
        "--full_prompt",
        generate_captions_args["full_prompt"],
        (
            "--use_nucleus_sampling"
            if find_boolean_value(generate_captions_args["use_nucleus_sampling"])
            else None
        ),
        "--num_captions",
        str(generate_captions_args["num_captions"]),
    ]
    generate_captions_command: List[str] = [
        cmd for cmd in _generate_captions_command if cmd is not None
    ]

    # Run generate_captions.py
    subprocess.run(generate_captions_command, check=True)

    # Arguments for clean_captions.py
    clean_captions_args = config["clean_captions"]
    clean_captions_command = [
        "python",
        "clean_captions.py",
        "--abs_path_to_objects",
        clean_captions_args["abs_path_to_objects"],
        "--abs_path_to_captions",
        clean_captions_args["abs_path_to_captions"],
        "--path_to_final_captions",
        clean_captions_args["path_to_final_captions"],
        "--openai_api_key",
        clean_captions_args["openai_api_key"] or os.environ["OPENAI_API_KEY"],
        "--gpt_type",
        clean_captions_args["gpt_type"],
        "--max_retries",
        str(clean_captions_args["max_retries"]),
        "--gpt_timeout",
        str(clean_captions_args["gpt_timeout"]),
        "--gpt_read_timeout",
        str(clean_captions_args["gpt_read_timeout"]),
        "--gpt_write_timeout",
        str(clean_captions_args["gpt_write_timeout"]),
        "--gpt_connect_timeout",
        str(clean_captions_args["gpt_connect_timeout"]),
        "--clip_model_name",
        clean_captions_args["clip_model_name"],
        "--prompt",
        clean_captions_args["prompt"],
    ]

    # Run clean_captions.py
    subprocess.run(clean_captions_command, check=True)


if __name__ == "__main__":
    main()

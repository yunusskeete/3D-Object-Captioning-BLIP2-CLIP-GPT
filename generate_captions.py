"""
Script to generate captions for images of objects using the Blip2T5 model.

This script processes images located in a specified directory structure, generates captions
for these images using a pretrained model, and saves the captions in a specified output directory.

Functions
---------
parse_args():
    Parses command-line arguments for the script.

caption_object_in_object_folder(model, eval_processor, object_folder_Path, **kwargs):
    Captions all images within a single object's folder.

caption_all_objects_in_parent_folder(model, eval_processor, **kwargs):
    Captions all objects in the parent folder.

main():
    Main function to run the script. Parses arguments, loads model, and captions objects.

Usage
-----
Run the script from the command line with the appropriate arguments. Example:
    python generate_captions.py --abs_path_to_objects /path/to/objects --output_dir /path/to/output --use_qa --use_nucleus_sampling --num_captions 4

Dependencies
------------
- argparse
- json
- pathlib
- typing
- torch
- lavis
- PIL
- tqdm
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Generator, List

import torch
from lavis.models import Blip2T5, load_model_and_preprocess
from lavis.processors.blip_processors import BlipImageEvalProcessor
from PIL import Image
from tqdm import tqdm

ENCODING = "utf-8"

# pylint: disable=C0103, W0718


def parse_args() -> argparse.Namespace:
    """
    Parses command-line arguments for the script.

    Returns
    -------
    argparse.Namespace
        An object containing the parsed arguments as attributes.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--abs_path_to_objects", type=str, default="object_renders")
    parser.add_argument("--output_dir", type=str, default="captions")
    parser.add_argument("--model_name", type=str, default="blip2_t5")
    parser.add_argument(
        "--model_type",
        type=str,
        default="pretrain_flant5xl",
        choices=["pretrain_flant5xxl", "pretrain_flant5xl"],
    )
    parser.add_argument("--use_qa", action="store_true")
    parser.add_argument(
        "--prompt", type=str, default="Question: what object is in this image? Answer:"
    )
    parser.add_argument(
        "--full_prompt",
        type=str,
        default="Question: what is the structure and geometry of this <object>?",
    )
    parser.add_argument("--use_nucleus_sampling", action="store_true")
    parser.add_argument("--num_captions", type=int, default=5)

    return parser.parse_args()


def caption_object_in_object_folder(
    model: Blip2T5,
    eval_processor: BlipImageEvalProcessor,
    object_folder_Path: Path,
    **kwargs,
) -> None:
    """
    Captions all images within a single object's folder.

    Parameters
    ----------
    model : Blip2T5
        The model to be used for generating captions.
    eval_processor : BlipImageEvalProcessor
        The evaluation processor for the images.
    object_folder_Path : Path
        The path to the folder containing object images.
    **kwargs
        Additional keyword arguments passed to the function, including:
        - output_dir (str): Directory where captions will be saved.
        - use_qa (bool): Whether to use question answering.
        - use_nucleus_sampling (bool): Whether to use nucleus sampling.
        - num_captions (int): Number of captions to generate per image.
        - prompt (str): The prompt template for generating captions.
        - full_prompt (str): The full prompt template for detailed captions.
        - device (torch.device): The device to run the model on (CPU or GPU).
    """
    print("Path to object renderings:", object_folder_Path)

    output_dir: str = kwargs.get("output_dir")
    use_qa: bool = kwargs.get("use_qa")
    use_nucleus_sampling: bool = kwargs.get("use_nucleus_sampling")
    num_captions: int = kwargs.get("num_captions")
    prompt: str = kwargs.get("prompt")
    full_prompt: str = kwargs.get("full_prompt")
    device: torch.device = kwargs.get("device")

    output_folder_Path: Path = Path(output_dir)
    output_folder_Path.mkdir(exist_ok=True)

    output_file_Path: Path = Path(
        output_folder_Path / f"{object_folder_Path.name}.json"
    )

    object_outputs: Dict[str, str] = {}

    if output_file_Path.exists():
        with output_file_Path.open("r", encoding=ENCODING) as f:
            object_outputs = json.load(f)

        print("Number of captions so far", len(object_outputs))

    object_images: List[Path] = sorted(object_folder_Path.rglob("*.png"))
    num_object_images: int = len(object_images)
    print("Number of PNG renders for this object:", num_object_images)

    if num_object_images == 0:
        return

    object_images = [
        img_Path for img_Path in object_images if img_Path.stem not in object_outputs
    ]
    num_images_to_caption = len(object_images)
    print("Number of images to caption:", num_images_to_caption)

    for image_name_Path in tqdm(
        object_images,
        total=num_images_to_caption,
        desc="Captioning images",
        unit="image",
    ):
        try:
            raw_image = Image.open(str(image_name_Path)).convert("RGB")
        except Exception as e:
            print(
                f"Could not open image, skipping. (Path to file: {image_name_Path}, Exception: {e})"
            )
            continue

        image: torch.Tensor = eval_processor(raw_image).unsqueeze(0).to(device)

        if use_qa:
            _object = model.generate({"image": image, "prompt": prompt})[0]
            _full_prompt: str = full_prompt.replace("<object>", _object)

            descriptions: List[str] = model.generate(
                {"image": image, "prompt": _full_prompt},
                use_nucleus_sampling=use_nucleus_sampling,
                num_captions=num_captions,
            )
        else:
            descriptions: List[str] = model.generate(
                {"image": image},
                use_nucleus_sampling=use_nucleus_sampling,
                num_captions=num_captions,
            )

        file_stem: str = image_name_Path.stem
        object_outputs[file_stem] = descriptions  # [desc for desc in descriptions]

    with output_file_Path.open("w", encoding=ENCODING) as f:
        json.dump(object_outputs, f, indent=4)


def caption_all_objects_in_parent_folder(
    model: Blip2T5, eval_processor: BlipImageEvalProcessor, **kwargs
) -> None:
    """
    Captions all objects in the parent folder.

    Parameters
    ----------
    model : Blip2T5
        The model to be used for generating captions.
    eval_processor : BlipImageEvalProcessor
        The evaluation processor for the images.
    **kwargs
        Additional keyword arguments passed to the function, including:
        - abs_path_to_objects (str): Absolute path to the directory containing objects.
    """
    abs_path_to_objects: str = kwargs.get("abs_path_to_objects")

    renderings_Path: Path = Path(abs_path_to_objects)
    assert (
        renderings_Path.exists()
    ), f"Path to renderings does not exist ({renderings_Path})"

    num_objects: int = sum(1 for item in renderings_Path.iterdir() if item.is_dir())
    print("Number of objects to caption:", num_objects)

    renderings_Path_generator: Generator[Path] = (
        item for item in renderings_Path.iterdir() if item.is_dir()
    )

    for object_idx, object_renderings_Path in enumerate(renderings_Path_generator):
        print(f"Captioning object {object_idx} of {num_objects}")

        caption_object_in_object_folder(
            model=model,
            eval_processor=eval_processor,
            object_folder_Path=object_renderings_Path,
            **kwargs,
        )


def main():
    """
    Main function to run the script.

    This function parses the command-line arguments, sets up the device,
    loads the model and its processors, and then captions all objects
    in the specified parent folder.
    """
    args: argparse.Namespace = parse_args()
    print("args:", args)
    args_dict: Dict[str, Any] = vars(args)

    ### Set up device
    device: torch.device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    args_dict["device"] = device

    ### Set up BLIP
    model, vis_processors, _ = load_model_and_preprocess(
        name=args.model_name, model_type=args.model_type, is_eval=True, device=device
    )
    eval_processor: BlipImageEvalProcessor = vis_processors["eval"]

    ### Caption objects
    caption_all_objects_in_parent_folder(
        model=model, eval_processor=eval_processor, **args_dict
    )


if __name__ == "__main__":
    main()

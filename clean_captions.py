"""
This script processes and summarizes descriptions for 3D object renderings using CLIP and GPT models.

The main steps of the script are as follows:
1. Parse command-line arguments to configure the script.
2. Set up the OpenAI API client and load the CLIP model.
3. Process each 3D object by:
   a. Loading captions and renderings.
   b. Calculating the similarity between each caption and its corresponding rendering using CLIP.
   c. Selecting the best caption for each view.
   d. Summarizing the selected captions into a final caption for each object using GPT.

Functions:
----------
- parse_args() -> argparse.Namespace:
    Parses command-line arguments and returns them as a namespace object.

- summarize_captions_gpt(descriptions: List[str], model: ["gpt-3.5-turbo", "gpt-4-turbo", "gpt-4o"] = "gpt-4o", top_p: float = 0.2, **kwargs) -> str:
    Summarizes a list of descriptions into a single concise caption using GPT.

- clean_captions(**kwargs) -> Dict[str, str]:
    Cleans and refines captions for all objects by selecting the best caption per view using cosine similarity and summarizing them using GPT.

- clean_captions_for_all_objects(all_objects_captions_Path: Path, renderings_Path: Path, **kwargs) -> Dict[str, str]:
    Cleans and refines captions for all objects by selecting the best caption per view using cosine similarity and summarizing them using GPT.

- clean_object_captions(object_captions: Dict[str, List[str]], object_renderings_views_Paths: Path, **kwargs) -> str:
    Cleans and refines captions for a single object by selecting the best caption per view using cosine similarity and summarizing them using GPT.

- main():
    Main function to run the script. Parses the command-line arguments, sets up the device, loads the model, and processes the captions to generate final captions for each object by cleaning and summarizing them.

Usage:
------
Run the script from the command line with the appropriate arguments. Example:
    python clean_captions.py --abs_path_to_objects /path/to/objects --abs_path_to_captions /path/to/captions --path_to_final_captions /path/to/output_captions --openai_api_key YOUR_API_KEY --gpt_type gpt-4o

Dependencies:
-------------
- argparse
- json
- os
- pathlib
- typing
- clip
- httpx
- openai
- PIL
- torch
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, List

import clip
import httpx
import openai
import PIL
import torch
from openai import OpenAI
from PIL import Image
from torch import Tensor
from torch.nn import CosineSimilarity

ENCODING = "utf-8"

# pylint: disable=C0103


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
    parser.add_argument("--abs_path_to_captions", type=str, default="captions")
    parser.add_argument(
        "--path_to_final_captions", type=str, default="final_captions.json"
    )
    parser.add_argument(
        "--openai_api_key", type=str, default=os.environ["OPENAI_API_KEY"]
    )
    parser.add_argument("--gpt_type", type=str, default="gpt-4o")
    parser.add_argument("--top_p", type=float, default=0.2)
    parser.add_argument("--max_retries", type=int, default=2)
    parser.add_argument("--gpt_timeout", type=float, default=60.0)
    parser.add_argument("--gpt_read_timeout", type=float, default=5.0)
    parser.add_argument("--gpt_write_timeout", type=float, default=10.0)
    parser.add_argument("--gpt_connect_timeout", type=float, default=2.0)
    parser.add_argument("--cosine_similarity_dim", type=int, default=1)
    parser.add_argument("--cosine_similarity_eps", type=int, default=1e-6)
    parser.add_argument("--clip_model_name", type=str, default="ViT-B/32")
    parser.add_argument(
        "--prompt",
        type=str,
        default="Given a set of descriptions about the same 3D object, distill these descriptions into one concise caption. The descriptions are as follows: <descriptions>. Avoid describing background, surface, and posture. The caption should be:",
    )

    return parser.parse_args()


def summarize_captions_gpt(
    descriptions: List[str],
    model: ["gpt-3.5-turbo", "gpt-4-turbo", "gpt-4o"] = "gpt-4o",
    top_p: float = 0.2,
    **kwargs,
) -> str:
    """
    Summarizes a list of descriptions into a single concise caption using GPT.

    Parameters
    ----------
    descriptions : List[str]
        A list of descriptions to be summarized.
    model : str, optional
        The GPT model to be used, by default "gpt-4o".
    top_p : float, optional
        The nucleus sampling parameter, by default 0.2.
    kwargs : dict
        Additional keyword arguments including the OpenAI client and prompt.

    Returns
    -------
    str
        The summarized caption.
    """
    client: openai.OpenAI = kwargs.get("client")
    prompt: str = kwargs.get("prompt")
    _prompt: str = prompt.replace(
        "<descriptions>", "'" + "', '".join(descriptions) + "'"
    )

    final_caption: str = ""
    try:
        response = client.chat.completions.create(
            model=kwargs.get("gpt_type", None) or model,
            messages=[{"role": "system", "content": _prompt}],
            top_p=kwargs.get("top_p", None) or top_p,
        )
        final_caption: str = (
            response.choices[0].message.content if response.choices else ""
        )
    except openai.APIConnectionError as e:
        print("The server could not be reached")
        print(e.__cause__)  # an underlying Exception, likely raised within httpx.
    except openai.RateLimitError as e:
        print("A 429 status code was received; we should back off a bit.")
        print(e)
    except openai.APIStatusError as e:
        print("Another non-200-range status code was received")
        print(e.status_code)
        print(e.response)

    return final_caption


def clean_captions(**kwargs) -> Dict[str, str]:
    """
    Cleans and refines captions for all objects by selecting the best caption per view
    using cosine similarity and summarizing them using GPT.

    Returns
    -------
    Dict[str, str]
        A dictionary mapping object names to their final cleaned captions.
    """
    abs_path_to_captions: str = kwargs.get("abs_path_to_captions")
    abs_path_to_objects: str = kwargs.get("abs_path_to_objects")

    all_objects_captions_Path: Path = Path(abs_path_to_captions)
    assert (
        all_objects_captions_Path.exists()
    ), f"Path to captions does not exist ({all_objects_captions_Path})"
    print("all_objects_captions_Path:", all_objects_captions_Path)

    num_captioned_objects: int = sum(1 for item in all_objects_captions_Path.iterdir())
    print("Number of captioned objects:", num_captioned_objects)

    renderings_Path: Path = Path(abs_path_to_objects)
    assert (
        renderings_Path.exists()
    ), f"Path to renderings does not exist ({renderings_Path})"
    print("renderings_Path:", renderings_Path)

    final_captions: Dict[str, str] = clean_captions_for_all_objects(
        all_objects_captions_Path=all_objects_captions_Path,
        renderings_Path=renderings_Path,
        **kwargs,
    )

    return final_captions


def clean_captions_for_all_objects(
    all_objects_captions_Path: Path, renderings_Path: Path, **kwargs
) -> Dict[str, str]:
    """
    Cleans and refines captions for all objects by selecting the best caption per view
    using cosine similarity and summarizing them using GPT.

    Parameters
    ----------
    all_objects_captions_Path : Path
        Path to the directory containing captions for all objects.
    renderings_Path : Path
        Path to the directory containing renderings of all objects.
    kwargs : dict
        Additional keyword arguments including device, model, preprocess, and cosine similarity.

    Returns
    -------
    Dict[str, str]
        A dictionary mapping object names to their final cleaned captions.
    """

    final_captions: Dict[str, str] = {}

    for object_captions_Path in all_objects_captions_Path.iterdir():
        object_stem: str = object_captions_Path.stem
        print("Object:", object_stem)

        object_renderings_Paths: List[Path] = [
            path for path in renderings_Path.iterdir() if path.name == object_stem
        ]
        if not object_renderings_Paths:
            print("No renderings for captioned object")
            continue

        object_renderings_Path: Path = object_renderings_Paths.pop()
        # print(object_renderings_Path.name)

        object_renderings_views_Paths: List[Path] = list(
            sorted(object_renderings_Path.rglob("*.png"))
        )
        # print([view.name for view in object_renderings_views_Paths])

        with object_captions_Path.open("r") as f:
            object_captions: Dict[str, List[str]] = json.load(f)

        final_caption: str = clean_object_captions(
            object_captions=object_captions,
            object_renderings_views_Paths=object_renderings_views_Paths,
            **kwargs,
        )

        assert (
            object_stem not in final_captions
        ), f"Objects must have unique names. Received {object_stem} more than once."
        final_captions[object_stem] = final_caption

    return final_captions


def clean_object_captions(
    object_captions: Dict[str, List[str]], object_renderings_views_Paths: Path, **kwargs
) -> str:
    """
    Cleans and refines captions for a single object by selecting the best caption per view
    using cosine similarity and summarizing them using GPT.

    Parameters
    ----------
    object_captions : Dict[str, List[str]]
        A dictionary mapping view names to a list of captions for that view.
    object_renderings_views_Paths : Path
        Paths to the rendering views for the object.
    kwargs : dict
        Additional keyword arguments including device, model, preprocess, and cosine similarity.

    Returns
    -------
    str
        The final cleaned caption for the object.
    """
    device: torch.device = kwargs.get("device")
    model: torch.nn.Module = kwargs.get("model")
    preprocess: Callable[[PIL.Image], torch.Tensor] = kwargs.get("preprocess")
    cos: torch.nn.CosineSimilarity = kwargs.get("cos")

    object_best_captions_per_view: List[str] = []

    # Loop through all views for this object
    for view_Path in object_renderings_views_Paths:
        view_stem: str = view_Path.stem
        # print(object_stem, view_stem)

        # Get the set of captions per view
        view_captions: List[str] = object_captions[view_stem]
        # Get the image for this view
        image_tensor: Tensor = (
            preprocess(Image.open(str(view_Path))).unsqueeze(0).to(device)
        )
        # Encode the set of captions
        # Encode the image
        text_embedding: Tensor = clip.tokenize(view_captions).to(device)
        with torch.no_grad():
            image_features = model.encode_image(image_tensor)
            text_features = model.encode_text(text_embedding)
        # Find the similarity between each caption and the image
        caption_image_similarity_scores = cos(image_features, text_features)
        # Keep the caption with the highest similarity to the image
        max_index: int = torch.argmax(caption_image_similarity_scores).item()
        best_caption_for_this_view: str = view_captions[max_index]
        object_best_captions_per_view.append(best_caption_for_this_view)

    unique_captions: List[str] = list(
        set(caption for caption in object_best_captions_per_view if caption)
    )
    # print("unique_captions:", unique_captions)

    final_caption: str
    match len(unique_captions):
        case 0:
            final_caption = ""
        case 1:
            final_caption = ""
        case _:
            final_caption = summarize_captions_gpt(
                descriptions=unique_captions, **kwargs
            )
    print("final_caption:", final_caption)

    return final_caption


def main():
    """
    Main function to run the script.

    This function parses the command-line arguments, sets up the device,
    loads the model, and processes the captions to generate final captions
    for each object by cleaning and summarizing them.
    """
    args: argparse.Namespace = parse_args()
    print("args:", args)
    args_dict: Dict[str, Any] = vars(args)

    ### Set up OpenAI API client
    client: openai.OpenAI = OpenAI(
        api_key=args.openai_api_key,
        max_retries=args.max_retries,
        timeout=httpx.Timeout(
            args.gpt_timeout,
            read=args.gpt_read_timeout,
            write=args.gpt_write_timeout,
            connect=args.gpt_connect_timeout,
        ),
    )
    args_dict["client"] = client

    ### Set up device
    device: torch.device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    args_dict["device"] = device

    # Set up CLIP
    cos: torch.nn.CosineSimilarity = CosineSimilarity(
        dim=args.cosine_similarity_dim, eps=args.cosine_similarity_eps
    )
    model, preprocess = clip.load(args.clip_model_name, device=device)
    args_dict["cos"] = cos
    args_dict["model"] = model
    args_dict["preprocess"] = preprocess

    path_to_final_captions: str = args.path_to_final_captions
    final_captions_Path: Path = Path(
        path_to_final_captions
    ).resolve()  # relative -> absolute path
    print("final_captions_Path:", final_captions_Path)

    final_captions: Dict[str, str] = clean_captions(**args_dict)

    with final_captions_Path.open("w", encoding=ENCODING) as f:
        json.dump(final_captions, f, indent=4)


if __name__ == "__main__":
    main()

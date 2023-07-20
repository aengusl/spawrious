# Import statements
from ast import parse
import os
from diffusers import StableDiffusionPipeline
from tqdm import tqdm
import itertools
from datetime import datetime
import argparse
from ml_collections import ConfigDict
import torch
import random
import numpy as np
from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer
from PIL import Image

# Get variables from command line
def get_config():
    # Define function to convert string to boolean
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    # Define argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--global_save_label", type=str, required=True, help="path to dataset folder")
    parser.add_argument("--batch_size", type=int, default=3, help="Batch size")
    parser.add_argument("--minibatch_size", type=int, default=4, help="Minibatch size")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda or cpu)")
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument("--num_iters", type=int, default=1, help="Number of balanced datasets to generate")
    parser.add_argument("--machine_name", type=str, default="default", help="Machine name - a label to distinguish between different machines")
    parser.add_argument("--animals_to_generate", type=str, nargs="+", default='all', help="List of animals to generate")
    parser.add_argument("--locations_to_generate", type=str, nargs="+", default='all', help="List of locations to generate")
    parser.add_argument("--locations_to_avoid", type=str, nargs="+", default=[], help="List of locations to avoid")

    config = ConfigDict(vars(parser.parse_args()))
    return config

# Define all necessary functions
def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

model = VisionEncoderDecoderModel.from_pretrained(
    "nlpconnect/vit-gpt2-image-captioning"
)
feature_extractor = ViTFeatureExtractor.from_pretrained(
    "nlpconnect/vit-gpt2-image-captioning"
)
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
max_length = 16
num_beams = 4
gen_kwargs = {"max_length": max_length, "num_beams": num_beams}

config = get_config()
global_save_label = config.global_save_label
batch_size = config.batch_size
minibatch_size = config.minibatch_size
device = config.device
seed = config.seed
num_iters = config.num_iters
machine_name = config.machine_name

animals_to_generate = config.animals_to_generate
locations_to_generate = config.locations_to_generate
locations_to_avoid = config.locations_to_avoid

now = datetime.now()
begin_exp_time = now.strftime("%d%b_%H%M%S")

# The model
pipe = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4", use_auth_token=True
)
pipe = pipe.to(device)

def predict_step(image_paths: list[str]) -> list[str]:
    images = []
    for image_path in image_paths:
        i_image = Image.open(image_path)
        if i_image.mode != "RGB":
            i_image = i_image.convert(mode="RGB")

        images.append(i_image)

    pixel_values = feature_extractor(images=images, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)

    output_ids = model.generate(pixel_values, **gen_kwargs)

    preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    preds = [pred.strip() for pred in preds]
    return preds

def caption_from_images(images: list) -> list[str]:
    """
    Generate captions from a list of images
    """
    pixel_values = feature_extractor(images=images, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)

    output_ids = model.generate(pixel_values, **gen_kwargs)

    preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    preds = [pred.strip() for pred in preds]
    return preds

def dirty_image_keyword_filter(images: list, keywords: list) -> bool:
    """
    Filter images by whether the caption contains the keywords for the object and background
    """
    dirty_bool = False
    preds = caption_from_images(images)
    for caption in preds:
        caption_words = caption.strip().split(" ")
        if not set(keywords) & set(caption_words):
            dirty_bool = True
            break
    return dirty_bool

def generate_batch(prompt: str, save_label: str, keywords: list = ['dog'], negative_prompt: str = "human, blurry, painting, cartoon, extra limbs, disfigured, deformed, body out of frame, bad anatomy, watermark, two, multiple", num_inference_steps: int = 150, batch_size: int = 3, minibatch_size: int = 4, additional_label: str = None):
    """
    Generate a batch of images from a prompt.
    Save the images to a folder specified by the save label, under the format
        <save_label>/<machine_name>_<idx>.png
    """
    print('Generating images with prompt: \n', prompt)
    os.makedirs(save_label, exist_ok=True)
    prompt_list = [prompt] * minibatch_size
    negative_prompt_list = [negative_prompt] * minibatch_size
    batch_count = 0
    cleaning_total = 0
    with tqdm(total=batch_size) as pbar:
        while batch_count < batch_size:
            # Generate a batch of images
            output = pipe(
                prompt_list,
                negative_prompt=negative_prompt_list,
                num_inference_steps=num_inference_steps,
            )
            images = output.images

            # Filter out dirty images
            dirty_bool = False
            nsfw_bool = sum(output.nsfw_content_detected) > 0 #True if any images are nsfw
            if dirty_bool or nsfw_bool:
                print('Bad images detected: \n', 'dirty_bool:', dirty_bool, ', nsfw_bool:', nsfw_bool)
                if dirty_bool:
                    cleaning_total += 1
                continue
            
            # Save the images
            for idx, image in enumerate(images):
                save_path = os.path.join(save_label, f"{machine_name}_{batch_count+idx}.png")
                if additional_label is not None:
                    save_path = os.path.join(save_label, f"{machine_name}_{additional_label}_{batch_count+idx}.png")
                image.save(save_path, format="png")
                pbar.update(1)
            batch_count += len(images)
    return cleaning_total


"""
Create prompt list dictionary of form:
    {'animal-background': [prompt1,..,prompD]}
"""

animal_list = [
    "labrador",
    "welsh corgi dog",
    "bulldog",
    "dachshund",
]
one_word_animal_list = [
    "labrador",
    "corgi",
    "bulldog",
    "dachshund",
]
animal_dict = {
    "labrador": "labrador",
    "corgi": "welsh corgi dog",
    "bulldog": "bulldog",
    "dachshund": "dachshund"
}
location_list = [
    'in a jungle',
    'on a rocky mountain',
    'in a hot, dry desert with cactuses around',
    'in a park, with puddles, bushes and dirt in the background',
    'playing fetch on a beach with a pier and ocean in the background',
    'in a snowy landscape with a cabin and a snowball in the background',
]
one_word_location_list = [
    'jungle',
    'mountain',
    'desert',
    'dirt',
    'beach',
    'snow',
]
location_dict = {
    'jungle': 'in a jungle',
    'mountain': 'on a rocky mountain',
    'desert': 'in a hot, dry desert with cactuses around',
    'dirt': 'in a park, with puddles, bushes and dirt in the background',
    'beach': 'playing fetch on a beach with a pier and ocean in the background',
    'snow': 'in a snowy landscape with a cabin and a snowball in the background',
}
fur_list = [
    "black",
    "brown",
    "white",
    "",
]
pose_list = [
    "sitting",
    "",
    "running",
]
tod_list = [
    "pale sunrise",
    "sunset",
    "rainy day",
    "foggy day",
    "bright sunny day",
    "bright sunny day",
]
prompt_template = "(((one {fur} {animal} {pose}))) {location}, {tod}. highly detailed, with cinematic lighting, 4k resolution, beautiful composition, hyperrealistic, trending, cinematic, masterpiece, close up"    

assert animals_to_generate in one_word_animal_list or animals_to_generate == 'all'
if animals_to_generate == 'all':
    pass
else:
    one_word_animal_list = [animals_to_generate]

assert locations_to_generate in one_word_location_list or locations_to_generate == 'all'
if locations_to_generate == 'all':
    pass
else:
    one_word_location_list = [locations_to_generate]

if locations_to_avoid != 'None':
    for loc in locations_to_avoid:
        one_word_location_list.remove(loc)

prompt_list_dict = {}
for animal_word in one_word_animal_list:
    for location_word in one_word_location_list:
        animal = animal_dict[animal_word]
        location = location_dict[location_word]
        prompt_list_dict[f'{animal_word}-{location_word}'] = []
        for fur in fur_list:
            for pose in pose_list:
                for tod in tod_list:
                    prompt = prompt_template.format(fur=fur, animal=animal, pose=pose, location=location, tod=tod)
                    prompt_list_dict[f'{animal_word}-{location_word}'].append(prompt)

# %%
"""
Generate a mini dataset with samples from each prompt
"""
for iteration in tqdm(range(num_iters)):
    print('\n\n\n\n\n\nIteration:', iteration, '\n\n\n\n\n\n')
    cleaning_total = 0
    for animal_loc in tqdm(prompt_list_dict.keys()):
        print('\n\n\nAnimal-Location:', animal_loc, '\n\n\n')
        prompt_count = 0
        for prompt in prompt_list_dict[animal_loc]:
            animal_str = animal_loc.split('-')[0]
            location_str = animal_loc.split('-')[1]
            save_label = os.path.join(global_save_label.format(iteration), f"{location_str}", f"{animal_str}")
            os.makedirs(save_label, exist_ok=True)
            cleaning_total += generate_batch(
                prompt=prompt,
                save_label=save_label,
                keywords = ['dog'],
                batch_size=batch_size,
                minibatch_size=minibatch_size,
                num_inference_steps=100,
                additional_label = f"prompt_{prompt_count}"
            )
            prompt_count += 1
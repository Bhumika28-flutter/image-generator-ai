import os
import sys
import uuid
import torch
import random
import logging
from textblob import TextBlob
from nltk.corpus import wordnet
import nltk
from diffusers import StableDiffusionPipeline

# First-time only: Download WordNet
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Dummy NSFW checker
def dummy_checker(images, **kwargs):
    return images, [False] * len(images)

# Load model
def load_model():
    try:
        logger.info("Loading model...")
        pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float32,
            use_safetensors=True,
            low_cpu_mem_usage=True
        )
        pipe.safety_checker = dummy_checker
        return pipe
    except Exception as e:
        logger.error(f"Model loading failed: {e}")
        raise

# Fix prompt spelling
def suggest_prompt(prompt):
    blob = TextBlob(prompt)
    corrected = str(blob.correct())
    return corrected if corrected != prompt else None

# Get synonyms using WordNet
def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            if '_' not in lemma.name():
                synonyms.add(lemma.name())
    return synonyms

# Generate related prompt variations
def generate_alternatives(base_prompt, max_alternatives=5):
    words = base_prompt.split()
    alternatives = [base_prompt]

    for i, word in enumerate(words):
        syns = get_synonyms(word)
        filtered = [s for s in syns if s.lower() != word.lower()]
        for syn in filtered[:2]:
            new_words = words.copy()
            new_words[i] = syn
            new_prompt = " ".join(new_words)
            if new_prompt not in alternatives:
                alternatives.append(new_prompt)
            if len(alternatives) >= max_alternatives:
                return alternatives[:max_alternatives]

    if len(alternatives) < max_alternatives:
        alternatives.append(base_prompt + " in beautiful style")
    if len(alternatives) < max_alternatives:
        alternatives.append(base_prompt + " detailed art")

    return alternatives[:max_alternatives]

# Generate image
def generate_images(pipe, prompt, num_images=1, seed=None, width=512, height=512, steps=20, scale=7.5, negative_prompt="blurry, low quality"):
    try:
        os.makedirs("images", exist_ok=True)

        if seed is None:
            seed = random.randint(0, 2**32 - 1)

        for i in range(num_images):
            image_seed = seed + i  # Unique seed for each image
            generator = torch.manual_seed(image_seed)
            logger.info(f"Generating image {i+1}/{num_images} | Seed: {image_seed}")

            with torch.no_grad():
                image = pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=steps,
                    guidance_scale=scale,
                    height=height,
                    width=width,
                    generator=generator
                ).images[0]

            filename = f"{uuid.uuid4().hex}.png"
            filepath = os.path.join("images", filename)
            image.save(filepath)
            print(f"✅ Generated: {filename} (Seed: {image_seed})")

        return True
    except Exception as e:
        logger.error(f"Batch generation error: {e}")
        return False

# Main execution
if __name__ == "__main__":
    pipe = load_model()

    if len(sys.argv) > 1:
        prompt = sys.argv[1]
    else:
        prompt = input("Enter prompt: ").strip()

    # Step 1: Spell check
    suggestion = suggest_prompt(prompt)
    if suggestion:
        print(f"\n💡 Did you mean: '{suggestion}' ?")
        use_suggestion = input("Use suggested prompt? (y/n): ").strip().lower()
        if use_suggestion == "y":
            prompt = suggestion

    # Step 2: Related prompt suggestions
    print("\n🧠 Related prompt suggestions:")
    alternatives = generate_alternatives(prompt)
    for i, alt in enumerate(alternatives):
        print(f"{i + 1}. {alt}")
    selected = input("Select a prompt number or press Enter to keep current: ").strip()
    if selected.isdigit() and 1 <= int(selected) <= len(alternatives):
        prompt = alternatives[int(selected) - 1]

    try:
        seed_input = input("Enter seed (or press Enter for random): ").strip()
        seed = int(seed_input) if seed_input else None

        width = int(input("Enter image width (default 512): ") or 512)
        height = int(input("Enter image height (default 512): ") or 512)
        steps = int(input("Enter steps (default 20): ") or 20)
        scale = float(input("Enter guidance scale (default 7.5): ") or 7.5)
        num_images = int(input("How many images to generate? (default 1): ") or 1)
    except Exception as e:
        logger.warning(f"Invalid input, using defaults: {e}")
        width, height, steps, scale, seed, num_images = 512, 512, 20, 7.5, None, 1

    success = generate_images(pipe, prompt, num_images, seed, width, height, steps, scale)
    sys.exit(0 if success else 1)












# import torch
# from diffusers import StableDiffusionPipeline
# from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer
# import os
# import uuid
# import sys
# import logging

# # Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# def load_model():
#     try:
#         logger.info("Loading model with CPU optimization...")
#         pipe = StableDiffusionPipeline.from_pretrained(
#             "runwayml/stable-diffusion-v1-5",
#             torch_dtype=torch.float32,  # Use float32 for CPU
#             use_safetensors=False,
#             safety_checker=None,  # 👈 disables the NSFW filter
#             low_cpu_mem_usage=True
#         )
#         return pipe
#     except Exception as e:
#         logger.error(f"Model loading failed: {str(e)}")
#         raise

# try:
#     pipe = load_model()
#     logger.info("Model loaded successfully")
# except Exception as e:
#     logger.error(f"Failed to initialize pipeline: {str(e)}")
#     sys.exit(1)

# def generate_image(prompt):
#     try:
#         os.makedirs("images", exist_ok=True)
#         filename = f"{uuid.uuid4().hex}.png"
#         filepath = os.path.join("images", filename)
        
#         logger.info(f"Generating: '{prompt}'")
        
#         # Reduce memory usage
#         with torch.no_grad():
#             image = pipe(
#                 prompt,
#                 num_inference_steps=20,  # Reduced from default 50
#                 guidance_scale=7.5,
#                 height=512,
#                 width=512
#             ).images[0]
        
#         image.save(filepath)
#         logger.info(f"Saved: {filename}")
#         print(filename)
#         return True
        
#     except Exception as e:
#         logger.error(f"Error: {str(e)}")
#         return False

# if __name__ == "__main__":
#     if len(sys.argv) < 2:
#         logger.error("No prompt provided")
#         sys.exit(1)
        
#     success = generate_image(sys.argv[1])
#     sys.exit(0 if success else 1)






    
# from PIL import Image
# import uuid

# # Dummy image generation for testing
# filename = f"{uuid.uuid4().hex}.png"
# filepath = f"images/{filename}"

# # Create a blank image (for testing)
# img = Image.new('RGB', (512, 512), color='green')
# img.save(filepath)

# print(filename)  # stdout to Node.js
    






# # generate.py
# import sys
# import time
# import uuid
# import os
# import torch
# from diffusers import StableDiffusionPipeline

# # Use a folder to store generated images
# output_dir = "images"
# os.makedirs(output_dir, exist_ok=True)

# # Use first argument as prompt
# prompt = sys.argv[1]

# # Load model
# pipe = StableDiffusionPipeline.from_pretrained(
#     "runwayml/stable-diffusion-v1-5",
#     torch_dtype=torch.float16,
# )
# pipe.to("cuda")
# pipe.enable_xformers_memory_efficient_attention()

# # Generate image
# image = pipe(prompt).images[0]

# # Generate unique filename
# filename = f"{int(time.time())}_{uuid.uuid4().hex}.png"
# filepath = os.path.join(output_dir, filename)

# # Save image
# image.save(filepath)

# # Return filename to Node.js
# print(filename)

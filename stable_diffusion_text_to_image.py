# -*- coding: utf-8 -*-
"""stable-Diffusion-text-to-image.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1t6LaoIf72jmXaQoUIYuag8K6A8RKIznd
"""

!pip install diffusers --upgrade

!pip install invisible_watermark transformers accelerate safetensors

from diffusers import DiffusionPipeline
import torch

# load both base & refiner


base = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
)
base.to("cuda")
refiner = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0",
    text_encoder_2=base.text_encoder_2,
    vae=base.vae,
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
)
refiner.to("cuda")

# Define how many steps and what % of steps to be run on each experts (80/20) here
n_steps = 40
high_noise_frac = 0.8

prompt = "A Perfume Advertise"

# run both experts
image = base(
    prompt=prompt,
    num_inference_steps=n_steps,
    denoising_end=high_noise_frac,
    output_type="latent",
).images
image = refiner(
    prompt=prompt,
    num_inference_steps=n_steps,
    denoising_start=high_noise_frac,
    image=image,
).images[0]

image


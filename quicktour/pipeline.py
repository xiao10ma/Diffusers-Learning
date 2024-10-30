import torch
from diffusers import DiffusionPipeline, EulerDiscreteScheduler, UNet2DModel, DDPMPipeline, DDPMScheduler

# Load a pretrained pipeline
dm_pretrained_path = "/data/duantong/mazipei/HuggingFace-Download-Accelerator/data/hf_cache/hub/models--stabilityai--stable-diffusion-2-1-base/snapshots/5ede9e4bf3e3fd1cb0ef2f7a3fff13ee514fdf06"
pipeline = DiffusionPipeline.from_pretrained(dm_pretrained_path, use_safetensors=True)
pipeline.to("cuda")
image = pipeline("A photo of a cat.").images[0]
image.save("cat1.png")

# swap schedulers
pipeline.scheduler = EulerDiscreteScheduler.from_config(pipeline.scheduler.config)
image = pipeline("A photo of a cat.").images[0]
image.save("cat2.png")
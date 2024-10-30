import torch
from diffusers import DiffusionPipeline, EulerDiscreteScheduler, UNet2DModel, DDPMPipeline, DDPMScheduler
import PIL.Image
import numpy as np
import tqdm

def display_sample(sample, i):
    image_processed = sample.cpu().permute(0, 2, 3, 1)
    image_processed = (image_processed + 1.0) * 127.5
    image_processed = image_processed.numpy().astype(np.uint8)
    image = PIL.Image.fromarray(image_processed[0])
    image.save(f"sample_{i}.png")

repo_id = "/data/duantong/mazipei/HuggingFace-Download-Accelerator/data/hf_cache/hub/models--google--ddpm-cat-256/snapshots/82ca0d5db4a5ec6ff0e9be8d86852490bc18a3d9"
model = UNet2DModel.from_pretrained(repo_id, use_safetensors=True)

torch.manual_seed(0)

noisy_sample = torch.randn(1, model.config.in_channels, model.config.sample_size, model.config.sample_size)

# Unlike a model, a scheduler does not have trainable weights.
scheduler = DDPMScheduler.from_pretrained(repo_id)
print(scheduler)

model.to("cuda")
noisy_sample = noisy_sample.to("cuda")

sample = noisy_sample

for i, t in enumerate(tqdm.tqdm(scheduler.timesteps)):
    print(t)
    # 1. predict the noise residual, based on the current sample and timestep
    with torch.no_grad():
        residual = model(sample, t).sample

    # 2. compute the previous noisy sample x_t -> x_t-1
    # based on the residual, timestep, and the current sample
    sample = scheduler.step(residual, t, sample).prev_sample

    # 3. optionally look at image
    if (i + 1) % 50 == 0:
        display_sample(sample, i + 1)
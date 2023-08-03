import io
import os
import uvicorn
import torch
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form
from starlette.responses import StreamingResponse, Response
from PIL import Image
from shap_e.diffusion.sample import sample_latents
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
from shap_e.models.download import load_model, load_config
from shap_e.util.notebooks import create_pan_cameras, decode_latent_images, gif_widget
from shap_e.util.image_util import load_image
from shap_e.util.notebooks import decode_latent_mesh

app = FastAPI()

# Pre load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
xm = load_model('transmitter', device=device)
model = load_model('image300M', device=device)
diffusion = diffusion_from_config(load_config('diffusion'))

def generate_3d_model(xm, latents, message_hash):
    save_folder = f'/data/shap_e/3dmodels{message_hash}'
    os.makedirs(save_folder, exist_ok=True)
    
    for i, latent in enumerate(latents):
        t = decode_latent_mesh(xm, latent).tri_mesh()
        with open(f'{save_folder}/mesh_{i}.ply', 'wb') as f:
            t.write_ply(f)
        with open(f'{save_folder}/mesh_{i}.obj', 'w') as f:
            t.write_obj(f)
            
@app.post("/generate_images")
async def generate_images(image: UploadFile = File(...), message_hash: Form[str] = None):
    
    batch_size = 4
    guidance_scale = 3.0
    
    # Load the image from UploadFile
    image_bytes = await image.read()
    image_pil = Image.open(io.BytesIO(image_bytes))
    image_pil = image_pil.resize((256, 256))

    # 写入文件 img_path = /data/shap_e/tmp/{message_hash}.png
    path = "/data/shap_e/tmp/"
    os.makedirs(path, exist_ok=True)
    img_path = f'{path}{message_hash}.png'
    image_pil.save(img_path)

    image_file = load_image(img_path)

    latents = sample_latents(
        batch_size=batch_size,
        model=model,
        diffusion=diffusion,
        guidance_scale=guidance_scale,
        model_kwargs=dict(images=[image_file] * batch_size),
        progress=True,
        clip_denoised=True,
        use_fp16=True,
        use_karras=True,
        karras_steps=64,
        sigma_min=1e-3,
        sigma_max=160,
        s_churn=0,
    )
    
    render_mode = 'nerf'  # you can change this to 'stf' for mesh rendering
    size = 64  # this is the size of the renders; higher values take longer to render.
    
    cameras = create_pan_cameras(size, device)
    images = []

    def generate():
        for i, latent in enumerate(latents):
            decoded_images = decode_latent_images(xm, latent, cameras, rendering_mode=render_mode)
            images.append(gif_widget(decoded_images))
    
    # Create a streaming response for the generated images

    # Call the generate_3d_model function
    generate_3d_model(xm, latents, message_hash)
    return Response("success")
    # return StreamingResponse(generate(), media_type="image/png")

@app.post("/test")
async def test(image: UploadFile = None, message_hash: str = None):
    batch_size = 4
    guidance_scale = 3.0

    # Load the image from UploadFile
    image_bytes = await image.read()
    image_pil = Image.open(io.BytesIO(image_bytes))
    image_pil = image_pil.resize((256, 256))
    image_array = np.array(image_pil)
    image_tensor = torch.tensor(image_array).permute(2, 0, 1).unsqueeze(0).to(device).float() / 255.0

    return "success"

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8091)



import io
from fastapi import FastAPI, UploadFile
from starlette.responses import StreamingResponse
from PIL import Image

app = FastAPI()

@app.post("/generate_images")
async def generate_images(image: UploadFile = None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    xm = load_model('transmitter', device=device)
    model = load_model('image300M', device=device)
    diffusion = diffusion_from_config(load_config('diffusion'))
    batch_size = 4
    guidance_scale = 3.0
    
    # Load the image from UploadFile
    image_bytes = await image.read()
    image_pil = Image.open(io.BytesIO(image_bytes))
    image_tensor = torch.tensor(np.array(image_pil)).permute(2, 0, 1).unsqueeze(0).to(device).float() / 255.0
    
    latents = sample_latents(
        batch_size=batch_size,
        model=model,
        diffusion=diffusion,
        guidance_scale=guidance_scale,
        model_kwargs=dict(images=[image_tensor] * batch_size),
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
    
    for i, latent in enumerate(latents):
        decoded_images = decode_latent_images(xm, latent, cameras, rendering_mode=render_mode)
        images.append(gif_widget(decoded_images))
    
    # Create a streaming response for the generated images
    def generate():
        for img in images:
            img_bytes = io.BytesIO()
            img.save(img_bytes, format='png')
            img_bytes.seek(0)
            yield img_bytes.getvalue()
    
    return StreamingResponse(generate(), media_type="image/png")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

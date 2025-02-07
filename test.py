import torch
import torch.nn as nn
import cv2
import numpy as np
from transformers import CLIPTokenizer, CLIPTextModel
from PIL import Image

# -----------------------------
# Define the LightweightGenerator architecture.
# (This must match the architecture used during training.)
# -----------------------------
class LightweightGenerator(nn.Module):
    def __init__(self, latent_dim=96, text_dim=768):
        super(LightweightGenerator, self).__init__()
        self.fc = nn.Linear(latent_dim + text_dim, 512 * 2 * 4 * 4)
        self.deconv = nn.Sequential(
            # Upsample both temporally and spatially:
            nn.ConvTranspose3d(512, 256, kernel_size=3, stride=2,
                               padding=1, output_padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(256, 128, kernel_size=3, stride=2,
                               padding=1, output_padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(128, 64, kernel_size=3, stride=2,
                               padding=1, output_padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            # Upsample spatially from 32 to 64 while keeping time dimension at 16:
            nn.ConvTranspose3d(64, 32, kernel_size=(1, 3, 3), stride=(1, 2, 2),
                               padding=(0, 1, 1), output_padding=(0, 1, 1)),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            # Final conv to produce 3 channels:
            nn.Conv3d(32, 3, kernel_size=3, padding=1),
            nn.Tanh()
        )
    
    def forward(self, z, text_emb):
        # Concatenate latent noise and text embedding.
        x = torch.cat([z, text_emb], dim=1)  # Shape: (B, latent_dim + text_dim)
        x = self.fc(x)                       # Shape: (B, 512*2*4*4)
        x = x.view(-1, 512, 2, 4, 4)          # Shape: (B, 512, 2, 4, 4)
        out = self.deconv(x)                 # Output: (B, 3, 16, 64, 64)
        return out

# -----------------------------
# Setup CLIP (Tokenizer and Text Model)
# -----------------------------
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
text_model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
TEXT_EMBED_DIM = text_model.config.hidden_size  # Typically 768

# Ensure the text model is moved to the proper device later.

@torch.no_grad()
def preprocess_text(captions, device="cpu"):
    """
    Given a list of caption strings, tokenize them and run them through the CLIP text model.
    This function explicitly moves every tensor to the specified device.
    Returns an averaged text embedding of shape (768,).
    """
    if len(captions) == 0:
        return torch.zeros(TEXT_EMBED_DIM, device=device)
    
    embeddings = []
    for cap in captions:
        inputs = tokenizer(cap, return_tensors="pt", padding=True, truncation=True)
        # Ensure every tensor in the dictionary is moved to the correct device:
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = text_model(**inputs)
        # Use the [CLS] token embedding (first token)
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # Shape: (1,768)
        embeddings.append(cls_embedding.squeeze(0))
    
    final_emb = torch.mean(torch.stack(embeddings), dim=0)
    return final_emb

# -----------------------------
# Video Generation Function
# -----------------------------
def generate_video(prompt, model_path=r"D:\projects\Video-generator\ai_video_generator.pth",
                   latent_dim=96, output_path="generated_video.mp4"):
    """
    Loads the saved generator, processes the text prompt, generates a video tensor
    (shape: (1, 3, 16, 64, 64)), and saves it as an MP4 file.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Move text model to device.
    text_model.to(device)
    text_model.eval()
    
    # Load the generator and its weights.
    generator = LightweightGenerator(latent_dim=latent_dim, text_dim=TEXT_EMBED_DIM).to(device)
    state_dict = torch.load(model_path, map_location=device)
    generator.load_state_dict(state_dict)
    generator.eval()
    
    # Process the text prompt into a text embedding.
    text_emb = preprocess_text([prompt], device=device)  # Shape: (768,)
    text_emb = text_emb.unsqueeze(0)  # Add batch dimension => (1, 768)
    
    # Generate latent noise.
    z = torch.randn(1, latent_dim, device=device)
    
    # Generate the video tensor.
    with torch.no_grad():
        video_tensor = generator(z, text_emb)  # (1, 3, 16, 64, 64)
    
    # Remove the batch dimension and move to CPU.
    video_tensor = video_tensor.squeeze(0).cpu()  # (3, 16, 64, 64)
    
    # The generator output is from Tanh, so values are in [-1,1]; normalize to [0,1].
    video_tensor = (video_tensor + 1) / 2.0
    video_tensor = video_tensor.clamp(0, 1).numpy()  # (3, 16, 64, 64)
    
    # Rearrange dimensions to get a list of frames: (16, 64, 64, 3)
    video_frames = np.transpose(video_tensor, (1, 2, 3, 0))
    video_frames = (video_frames * 255).astype(np.uint8)
    
    # Save the video using OpenCV.
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 4  # Adjust playback speed as desired.
    height, width, _ = video_frames[0].shape
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for frame in video_frames:
        # Convert from RGB to BGR (OpenCV uses BGR)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)
    out.release()
    
    print(f"Video saved to {output_path}")
    return video_frames

# -----------------------------
# Main testing block
# -----------------------------
if __name__ == "__main__":
    prompt = input("Enter your text prompt: ")
    generate_video(prompt)

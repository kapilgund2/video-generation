import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import CyclicLR
from torch.nn.utils import spectral_norm

import numpy as np
import csv
import os
import cv2
from PIL import Image

# For data augmentations
from torchvision.transforms import Compose, Resize, Normalize, ToTensor, RandomHorizontalFlip

# For CLIP text embeddings
from transformers import CLIPTokenizer, CLIPTextModel
from tqdm import tqdm

###############################################################################
# 1. CSV & Directory Paths
###############################################################################
csv_file = r"D:\projects\Video-generator\openvid-1m-filtered.csv"
video_dir = r"D:\projects\Video-generator\download\OpenVid_part0\OpenVid_part1"

all_pairs = []
with open(csv_file, mode="r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        video_filename = row["video"].strip()
        caption = row["caption"]
        video_path = os.path.join(video_dir, video_filename)
        if os.path.exists(video_path):
            all_pairs.append((video_filename, caption))

print(f"Total CSV lines used (video files found): {len(all_pairs)}")

###############################################################################
# 2. Video Preprocessing (Always 16 Frames)
###############################################################################
def make_video_transform():
    return Compose([
        RandomHorizontalFlip(p=0.5),
        Resize((64, 64)),
        ToTensor(),
        Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
    ])

def load_video_frames(video_path, num_frames, transform):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return []
    step = max(total_frames // num_frames, 1)
    frames = []
    for i in range(num_frames):
        frame_idx = i * step
        if frame_idx >= total_frames:
            break
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)
        frame = transform(frame)  # (3,64,64)
        frames.append(frame)
    cap.release()
    return frames

def preprocess_video(video_path, transform, num_frames=16):
    frames = load_video_frames(video_path, num_frames, transform)
    if len(frames) == 0:
        return torch.zeros((3, num_frames, 64, 64))
    if len(frames) < num_frames:
        num_missing = num_frames - len(frames)
        pad = torch.zeros((num_missing, *frames[0].shape))
        frames = torch.stack(frames, dim=0)
        frames = torch.cat([frames, pad], dim=0)
    else:
        frames = torch.stack(frames[:num_frames], dim=0)
    # Permute from (16,3,64,64) to (3,16,64,64)
    frames = frames.permute(1, 0, 2, 3)
    return frames

###############################################################################
# 3. CLIP Text Embeddings
###############################################################################
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
text_model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
TEXT_EMBED_DIM = text_model.config.hidden_size  # typically 768

@torch.no_grad()
def preprocess_text(captions, device="cpu"):
    if len(captions) == 0:
        return torch.zeros(TEXT_EMBED_DIM, device=device)
    embeddings = []
    for cap in captions:
        inputs = tokenizer(cap, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = text_model(**inputs)
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # (1,768)
        embeddings.append(cls_embedding.squeeze(0))
    final_emb = torch.mean(torch.stack(embeddings), dim=0)
    return final_emb

###############################################################################
# 4. Video Dataset (Always 16 Frames)
###############################################################################
class VideoDataset(Dataset):
    def __init__(self, csv_pairs, video_dir, transform):
        self.csv_pairs = csv_pairs
        self.video_dir = video_dir
        self.transform = transform
        self.num_frames = 16
    def __len__(self):
        return len(self.csv_pairs)
    def __getitem__(self, idx):
        video_filename, caption_str = self.csv_pairs[idx]
        video_path = os.path.join(self.video_dir, video_filename)
        video_tensor = preprocess_video(video_path, self.transform, num_frames=16)
        return caption_str, video_tensor

###############################################################################
# 5. Residual Block for 3D Convolutions
###############################################################################
class ResidualBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock3D, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.skip = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm3d(out_channels)
            )
    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.skip(x)
        out = self.relu(out)
        return out

###############################################################################
# 6. Deeper Generator (with Residual Blocks)
###############################################################################
class DeeperGenerator(nn.Module):
    def __init__(self, latent_dim=96, text_dim=768):
        super(DeeperGenerator, self).__init__()
        self.fc = nn.Linear(latent_dim + text_dim, 512 * 2 * 4 * 4)
        # Initial upsampling: from (B,512,2,4,4) to (B,512,4,8,8)
        self.deconv_initial = nn.Sequential(
            nn.ConvTranspose3d(512, 512, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm3d(512),
            nn.ReLU(inplace=True)
        )
        self.resblock1 = ResidualBlock3D(512, 512, stride=1)
        # Further upsampling: (B,512,4,8,8) -> (B,256,8,16,16)
        self.deconv_mid = nn.Sequential(
            nn.ConvTranspose3d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True)
        )
        self.resblock2 = ResidualBlock3D(256, 256, stride=1)
        # Further upsampling: (B,256,8,16,16) -> (B,128,16,32,32)
        self.deconv_late = nn.Sequential(
            nn.ConvTranspose3d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True)
        )
        self.resblock3 = ResidualBlock3D(128, 128, stride=1)
        # Upsample spatially only from 32 to 64, keeping time=16:
        self.deconv_spatial = nn.Sequential(
            nn.ConvTranspose3d(128, 64, kernel_size=(1,3,3), stride=(1,2,2),
                               padding=(0,1,1), output_padding=(0,1,1)),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True)
        )
        self.final_conv = nn.Sequential(
            nn.Conv3d(64, 3, kernel_size=3, padding=1),
            nn.Tanh()
        )
    def forward(self, z, text_emb):
        x = torch.cat([z, text_emb], dim=1)  # (B, latent_dim + text_dim)
        x = self.fc(x)                       # (B, 512*2*4*4)
        x = x.view(-1, 512, 2, 4, 4)          # (B,512,2,4,4)
        x = self.deconv_initial(x)           # (B,512,4,8,8)
        x = self.resblock1(x)
        x = self.deconv_mid(x)               # (B,256,8,16,16)
        x = self.resblock2(x)
        x = self.deconv_late(x)              # (B,128,16,32,32)
        x = self.resblock3(x)
        x = self.deconv_spatial(x)           # (B,64,16,64,64)
        out = self.final_conv(x)             # (B,3,16,64,64)
        return out

###############################################################################
# 7. Deeper Discriminator (with Residual Blocks)
###############################################################################
class DeeperDiscriminator(nn.Module):
    def __init__(self):
        super(DeeperDiscriminator, self).__init__()
        self.conv_initial = nn.Sequential(
            spectral_norm(nn.Conv3d(3, 64, kernel_size=3, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.resblock1 = ResidualBlock3D(64, 128, stride=2)
        self.resblock2 = ResidualBlock3D(128, 256, stride=2)
        self.resblock3 = ResidualBlock3D(256, 512, stride=2)
        self.adaptive_pool = nn.AdaptiveAvgPool3d((1,4,4))
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 1 * 4 * 4, 1)
        )
    def forward(self, x):
        x = self.conv_initial(x)
        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.resblock3(x)
        x = self.adaptive_pool(x)
        out = self.fc(x)
        return out

###############################################################################
# 8. WGAN-GP Helper
###############################################################################
def gradient_penalty(discriminator, real_samples, fake_samples, device="cpu", gp_lambda=10.0):
    batch_size = real_samples.size(0)
    alpha = torch.rand(batch_size, 1, 1, 1, 1, device=device)
    interpolates = alpha * real_samples + (1 - alpha) * fake_samples
    interpolates.requires_grad_(True)
    d_interpolates = discriminator(interpolates)
    grad_outputs = torch.ones_like(d_interpolates, device=device)
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True
    )[0]
    gradients = gradients.view(batch_size, -1)
    grad_norm = gradients.norm(2, dim=1)
    penalty = torch.mean((grad_norm - 1) ** 2)
    return gp_lambda * penalty

###############################################################################
# 9. collate_fn + Training Loop
###############################################################################
def collate_fn(batch):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    caption_strs = [item[0] for item in batch]
    video_tensors = [item[1] for item in batch]
    video_tensors = torch.stack(video_tensors, dim=0)  # (B,3,16,64,64)
    with torch.no_grad():
        text_embs = []
        for cap_str in caption_strs:
            emb = preprocess_text([cap_str], device=device)
            text_embs.append(emb)
        text_embs = torch.stack(text_embs, dim=0)  # (B,768)
    return text_embs, video_tensors

def train_deeper_gan(
    csv_pairs, video_dir, num_epochs=10, batch_size=4,
    latent_dim=96, lr=0.0002, gp_lambda=10.0, accum_steps=4
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    text_model.to(device)
    text_model.eval()
    
    generator = DeeperGenerator(latent_dim=latent_dim, text_dim=TEXT_EMBED_DIM).to(device)
    discriminator = DeeperDiscriminator().to(device)
    
    optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.5,0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5,0.999))
    
    scheduler_G = CyclicLR(optimizer_G, base_lr=lr/2, max_lr=lr*2,
                           step_size_up=100, cycle_momentum=False)
    scheduler_D = CyclicLR(optimizer_D, base_lr=lr/2, max_lr=lr*2,
                           step_size_up=100, cycle_momentum=False)
    
    scaler_G = torch.amp.GradScaler(enabled=(device.type=='cuda'))
    scaler_D = torch.amp.GradScaler(enabled=(device.type=='cuda'))
    
    transform = make_video_transform()
    dataset = VideoDataset(csv_pairs, video_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                            collate_fn=collate_fn, drop_last=True)
    
    for epoch in range(num_epochs):
        print(f"\n=== Epoch [{epoch+1}/{num_epochs}] ===")
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        optimizer_D.zero_grad(set_to_none=True)
        optimizer_G.zero_grad(set_to_none=True)
        for i, (text_emb, real_videos) in enumerate(pbar):
            b_size = real_videos.size(0)
            text_emb = text_emb.to(device)
            real_videos = real_videos.to(device)
            
            # Train Discriminator
            for param in discriminator.parameters():
                param.requires_grad = True
            with torch.amp.autocast(device_type='cuda', enabled=(device.type=='cuda')):
                z = torch.randn(b_size, latent_dim, device=device)
                fake_videos = generator(z, text_emb)
                d_real = discriminator(real_videos)
                d_fake = discriminator(fake_videos.detach())
                d_loss = torch.mean(d_fake) - torch.mean(d_real)
                gp_val = gradient_penalty(discriminator, real_videos, fake_videos.detach(), device, gp_lambda)
                d_loss_total = d_loss + gp_val
            scaler_D.scale(d_loss_total).backward()
            if ((i+1) % accum_steps == 0) or (i+1 == len(dataloader)):
                scaler_D.step(optimizer_D)
                scaler_D.update()
                optimizer_D.zero_grad(set_to_none=True)
                scheduler_D.step()
            
            # Train Generator (update every n_critic steps)
            if (i+1) % 5 == 0:
                for param in discriminator.parameters():
                    param.requires_grad = False
                with torch.amp.autocast(device_type='cuda', enabled=(device.type=='cuda')):
                    z = torch.randn(b_size, latent_dim, device=device)
                    gen_videos = generator(z, text_emb)
                    d_gen = discriminator(gen_videos)
                    g_loss = -torch.mean(d_gen)
                scaler_G.scale(g_loss).backward()
                if ((i+1) % accum_steps == 0) or (i+1 == len(dataloader)):
                    scaler_G.step(optimizer_G)
                    scaler_G.update()
                    optimizer_G.zero_grad(set_to_none=True)
                    scheduler_G.step()
            
            pbar.set_postfix({
                "D_loss": f"{d_loss.item():.4f}",
                "GP": f"{gp_val.item():.4f}",
                "G_loss": f"{g_loss.item():.4f}" if (i+1)%5==0 else "N/A"
            })
    return generator, discriminator

###############################################################################
# 10. Main Script
###############################################################################
if __name__ == "__main__":
    num_epochs = 10
    batch_size = 4
    latent_dim = 96
    lr = 0.0002
    gp_lambda = 10.0

    print(f"Final dataset size: {len(all_pairs)}")
    generator, discriminator = train_deeper_gan(all_pairs, video_dir, num_epochs, batch_size,
                                                 latent_dim, lr, gp_lambda, accum_steps=4)
    torch.save(generator.state_dict(), "deeper_video_generator.pth")
    print("Saved deeper generator to 'deeper_video_generator.pth'.")

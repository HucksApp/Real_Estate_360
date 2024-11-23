import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms, models
import imageio
from tqdm import tqdm
import numpy as np
from PIL import Image
import open3d as o3d

# NeRF Network Definition with Camera Ray Directions


class NeRF(nn.Module):
    def __init__(self, depth=8, width=256):
        super(NeRF, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(depth):
            self.layers.append(
                nn.Linear(width + 3 if i == 0 else width,
                          width)  # Add direction input
            )
            self.layers.append(nn.ReLU())
        self.final_layer = nn.Linear(width, 4)

    def forward(self, x, view_dir=None):
        # Concatenate position and viewing direction
        if view_dir is not None:
            x = torch.cat([x, view_dir], dim=-1)
        for layer in self.layers:
            x = layer(x)
        return self.final_layer(x)

# Generator and Discriminator for Pix2Pix3D


class Generator(nn.Module):
    def __init__(self, input_channels, output_channels, attention=False):
        super(Generator, self).__init__()
        self.down = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(64) for _ in range(9)]
        )
        self.attention = AttentionBlock(64) if attention else nn.Identity()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(64, output_channels,
                               kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.down(x)
        x = self.residual_blocks(x)
        x = self.attention(x)
        return self.up(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
        )

    def forward(self, x):
        return x + self.conv_block(x)


class AttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super(AttentionBlock, self).__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 8, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 8, in_channels, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.attention(x)


# Pix2Pix3D Dataset
class Pix2Pix3DDataset(Dataset):
    def __init__(self, image_dir, target_dir, pose_dir, transform=None):
        if not os.path.exists(image_dir):
            raise FileNotFoundError(f"Image directory {image_dir} not found.")
        if not os.path.exists(target_dir):
            raise FileNotFoundError(
                f"Target directory {target_dir} not found.")
        if not os.path.exists(pose_dir):
            raise FileNotFoundError(f"Pose directory {pose_dir} not found.")
        self.image_dir = image_dir
        self.target_dir = target_dir
        self.pose_dir = pose_dir
        self.transform = transform
        self.image_filenames = [f for f in os.listdir(
            image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_filenames[idx])
        target_path = os.path.join(self.target_dir, self.image_filenames[idx])
        pose_path = os.path.join(
            self.pose_dir, self.image_filenames[idx].replace(".png", ".txt"))
        image = Image.open(image_path).convert("RGB")
        target = Image.open(target_path).convert("RGB")

        # Load pose as a 4x4 matrix
        pose = np.loadtxt(pose_path).astype(
            np.float32)  # Ensure it's a float32 matrix

        if self.transform:
            image = self.transform(image)
            target = self.transform(target)

        return image, target, torch.tensor(pose)


# Combined NeRF + Pix2Pix3D Model for 3D Generation from Multiple Views
class NeRF_Pix2Pix3D_Model(nn.Module):
    def __init__(self, title, dataset_path, save_dir, training=True):
        super(NeRF_Pix2Pix3D_Model, self).__init__()
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.title = title
        self.save_dir = os.path.join(save_dir, title)
        os.makedirs(self.save_dir, exist_ok=True)
        self.generator = Generator(
            input_channels=4, output_channels=3, attention=True).to(self.device)
        self.nerf_model = NeRF().to(self.device)

        # Load pre-trained VGG model for perceptual loss
        self.vgg = models.vgg16(
            pretrained=True).features.to(self.device).eval()

        # Freeze VGG model layers for perceptual loss
        for param in self.vgg.parameters():
            param.requires_grad = False

        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.RandomApply([
                transforms.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
            ], p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        if training:
            self.dataset = Pix2Pix3DDataset(image_dir=os.path.join(dataset_path, 'images'),
                                            target_dir=os.path.join(
                                                dataset_path, 'targets'),
                                            pose_dir=os.path.join(
                                                dataset_path, 'poses'),
                                            transform=self.transform)
            self.dataloader = DataLoader(
                self.dataset, batch_size=16, shuffle=True)
            self.optimizer_G = optim.Adam(
                self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
            self.optimizer_N = optim.Adam(
                self.nerf_model.parameters(), lr=5e-4)
            self.l1_loss = nn.L1Loss()
            self.mse_loss = nn.MSELoss()
        else:
            self.load_model()

    def ray_march(self, nerf_model, camera_pos, camera_dir, num_samples=64, near=0.1, far=4.0):
        # Optimized ray marching with batch processing
        t_vals = torch.linspace(near, far, num_samples).to(
            self.device)  # Parametric values along the ray
        ray_positions = camera_pos + \
            camera_dir.unsqueeze(0) * t_vals.unsqueeze(1)  # (num_samples, 3)

        # Use batched processing for ray colors and densities
        ray_colors, ray_densities = [], []
        for pos in ray_positions:
            # NeRF model outputs color and density
            color, density = nerf_model(pos)
            ray_colors.append(color)
            ray_densities.append(density)

        ray_colors = torch.stack(ray_colors, dim=0)  # (num_samples, 3)
        ray_densities = torch.stack(ray_densities, dim=0)  # (num_samples, 1)

        # Volume rendering
        color_output = torch.zeros(3).to(self.device)
        opacity_accum = 0.0

        for i in range(num_samples):
            alpha = 1.0 - torch.exp(-ray_densities[i])  # Opacity
            color_output += (1.0 - opacity_accum) * \
                ray_colors[i] * alpha  # Accumulate color
            opacity_accum += alpha

        return color_output  # RGB value of the rendered pixel

    def perceptual_loss(self, generated_image, target_image):
        # Compute perceptual loss using VGG
        generated_features = self.vgg(generated_image)
        target_features = self.vgg(target_image)
        loss = self.mse_loss(generated_features, target_features)
        return loss

    def render_scene_with_pose(self, nerf_model, pose, image_size=(512, 512), num_samples=64):
        camera_pos = pose[:3, 3]  # Camera position from pose
        rotation_matrix = pose[:3, :3]  # Rotation matrix from pose
        rays = []

        for i in range(image_size[0]):
            for j in range(image_size[1]):
                # Convert pixel coordinates to camera space
                pixel_coords = torch.tensor(
                    [i, j], dtype=torch.float32).unsqueeze(0).to(self.device)
                ray_dir = self.get_ray_direction(
                    pixel_coords, image_size, rotation_matrix)
                rays.append(ray_dir)

        rendered_image = self.ray_march(
            nerf_model, camera_pos, rays, num_samples=num_samples)
        return rendered_image

    def get_ray_direction(self, pixel_coords, image_size, rotation_matrix):
        # Convert pixel coordinates to normalized camera space rays
        fx, fy = image_size[0] / 2.0, image_size[1] / 2.0
        cx, cy = image_size[0] / 2.0, image_size[1] / 2.0

        # Normalize pixel coordinates
        normalized_coords = (pixel_coords - torch.tensor([cx, cy])) / \
            torch.tensor([fx, fy])

        ray_dir = rotation_matrix @ normalized_coords.t()
        return ray_dir

    def forward(self, batch):
        # Forward pass for training or inference
        images, targets, poses = batch
        rendered_image = self.render_scene_with_pose(
            self.nerf_model, poses, num_samples=64)
        generated_image = self.generator(rendered_image)
        loss = self.l1_loss(generated_image, targets)
        return generated_image, loss

    def style_loss(generated_image, target_image, vgg):
        # Extract features from intermediate layers of VGG
        generated_features = vgg(generated_image)
        target_features = vgg(target_image)

        # Compute the mean squared error of the feature maps (style loss)
        loss = nn.MSELoss()(generated_features, target_features)
        return loss

    def cycle_consistency_loss(original_image, reconstructed_image):
        # Compute L1 loss for cycle consistency
        loss = nn.L1Loss()(original_image, reconstructed_image)
        return loss

    def train_step(self, batch):
        images, targets, poses = batch
        images = images.to(self.device)
        targets = targets.to(self.device)

        self.optimizer_G.zero_grad()
        self.optimizer_N.zero_grad()

        # Forward pass through the generator
        generated_images = self.generator(images)

        # Render the scene using the NeRF model (using the pose to get camera position and direction)
        rendered_images = self.render_scene_with_pose(
            self.nerf_model, poses[0].cpu().numpy())

        # Loss computation
        # L1 loss between generated and target images
        loss_img = self.l1_loss(generated_images, targets)
        # MSE loss between rendered and target images
        loss_nerf = self.mse_loss(rendered_images, targets)
        loss_perceptual = self.perceptual_loss(
            generated_images, targets)  # Perceptual loss using VGG
        loss_style = self.style_loss(
            generated_images, targets, self.vgg)  # StyleGAN-style loss
        # Forward pass to compute cycle consistency
        cycle_image = self.generator(generated_images)
        loss_cycle = self.cycle_consistency_loss(
            images, cycle_image)  # Cycle consistency loss

        # Total loss: combining all losses
        total_loss = loss_img + loss_nerf + loss_perceptual + loss_style + loss_cycle
        total_loss.backward()

        # Optimizer steps
        self.optimizer_G.step()
        self.optimizer_N.step()

        return total_loss

    def train(self, num_epochs=50):
        for epoch in range(num_epochs):
            total_loss = 0
            for batch in tqdm(self.dataloader):
                loss = self.train_step(batch)
                total_loss += loss.item()
            print(
                f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(self.dataloader)}")
            self.save_model(epoch)

    def load_model(self):
        # Load model weights if exists
        generator_path = os.path.join(
            self.save_dir, "generator_epoch_last.pth")
        nerf_model_path = os.path.join(
            self.save_dir, "nerf_model_epoch_last.pth")

        if os.path.exists(generator_path):
            self.generator.load_state_dict(torch.load(generator_path))
            print(f"Generator model loaded from {generator_path}")
        if os.path.exists(nerf_model_path):
            self.nerf_model.load_state_dict(torch.load(nerf_model_path))
            print(f"NeRF model loaded from {nerf_model_path}")

    def save_model(self, epoch):
        torch.save(self.generator.state_dict(), os.path.join(
            self.save_dir, f"generator_epoch_{epoch+1}.pth"))
        torch.save(self.nerf_model.state_dict(), os.path.join(
            self.save_dir, f"nerf_model_epoch_{epoch+1}.pth"))

    def save_model_for_inference(self):
        # Save the complete model (architecture + weights) for inference in other environments
        torch.save({
            'model_state_dict': self.state_dict(),
            'generator_state_dict': self.generator.state_dict(),
            'nerf_model_state_dict': self.nerf_model.state_dict()
        }, os.path.join(self.save_dir, 'full_model.pth'))

    def render(self, pose):
        self.nerf_model.eval()
        with torch.no_grad():
            rendered_image = self.nerf_model(
                torch.tensor(pose).to(self.device))
        return rendered_image

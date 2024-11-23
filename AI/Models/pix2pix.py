import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torch import nn
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import open3d as o3d
import numpy as np
from PIL import Image


# Dataset Class for Pix2Pix3D
class Pix2Pix3DDataset(torch.utils.data.Dataset):
    """
    Dataset class for Pix2Pix3D model. Manages paired image and target datasets.
    """

    def __init__(self, image_dir, target_dir, transform=None):
        if not os.path.exists(image_dir):
            raise FileNotFoundError(f"Image directory {image_dir} not found.")
        if not os.path.exists(target_dir):
            raise FileNotFoundError(
                f"Target directory {target_dir} not found.")

        self.image_dir = image_dir
        self.target_dir = target_dir
        self.transform = transform
        self.image_filenames = [f for f in os.listdir(
            image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_filenames[idx])
        target_path = os.path.join(self.target_dir, self.image_filenames[idx])

        image = Image.open(image_path).convert("RGB")
        target = Image.open(target_path).convert("RGB")

        if self.transform:
            image = self.transform(image)
            target = self.transform(target)

        return image, target


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


class Discriminator(nn.Module):
    def __init__(self, input_channels):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 1, kernel_size=4, stride=1, padding=1),
        )

    def forward(self, x):
        return self.model(x)


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

# Pix2Pix3DModel


class Pix2Pix3DModel(nn.Module):
    def __init__(self, title, training=False, dataset_path=None, save_dir=None):
        super(Pix2Pix3DModel, self).__init__()

        if save_dir is None:
            raise ValueError("save_dir must be provided.")

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.title = title
        self.save_dir = os.path.join(save_dir, title)
        os.makedirs(self.save_dir, exist_ok=True)

        # Generator and discriminator initialization
        self.generator = Generator(
            input_channels=4, output_channels=3, attention=True).to(self.device)
        self.discriminator = Discriminator(input_channels=3).to(
            self.device) if training else None

        # Load weights for inference or training continuation
        self.load_model_weights()

        # Only set up dataset and transforms for training
        if training:
            if dataset_path is None:
                raise ValueError(
                    "dataset_path must be provided in training mode.")

            self.transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
            self.depth_model = torch.hub.load(
                "intel-isl/MiDaS", "MiDaS_small").to(self.device)
            self.midas_transforms = torch.hub.load(
                "intel-isl/MiDaS", "transforms").default_transform
            self.dataset = Pix2Pix3DDataset(
                dataset_path, dataset_path, transform=self.transform)
            self.dataloader = DataLoader(
                self.dataset, batch_size=16, shuffle=True)
            self.midas_transforms = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ])

            # Configure training
            self._configure_training()

    def load_model_weights(self):
        """Load previously saved model weights for inference or continuation of training."""
        generator_path = os.path.join(self.save_dir, "generator.pth")
        discriminator_path = os.path.join(self.save_dir, "discriminator.pth")

        if os.path.exists(generator_path):
            print(f"Loading generator weights from {generator_path}")
            self.generator.load_state_dict(torch.load(generator_path))
        else:
            print("No generator weights found, using random initialization.")

        if os.path.exists(discriminator_path):
            print(f"Loading discriminator weights from {discriminator_path}")
            self.discriminator.load_state_dict(torch.load(discriminator_path))
        else:
            print("No discriminator weights found, using random initialization.")

    def preprocess_image(self, image_path):
        """Preprocess input image and generate depth map."""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Input image {image_path} not found.")

        image = Image.open(image_path).convert("RGB")

        # Prepare image for MiDaS
        midas_input = self.midas_transforms(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            depth_map = self.depth_model(midas_input)
            depth_map = torch.nn.functional.interpolate(
                depth_map, size=(256, 256), mode="bicubic", align_corners=False)

        # Prepare RGB image
        rgb_image = self.transform(image).unsqueeze(0).to(self.device)

        # Concatenate RGB and depth map along the channel dimension
        combined_input = torch.cat((rgb_image, depth_map), dim=1)
        return combined_input

    def _configure_training(self):
        """Setup for training mode, including adversarial loss."""
        self.generator.train()
        self.discriminator = Discriminator(input_channels=3).to(
            self.device)  # Discriminator model
        self.discriminator.train()
        self.depth_model.eval()  # Pre-trained depth model remains in eval mode

        # Define loss functions
        self.l1_loss = nn.L1Loss()
        self.adversarial_loss = nn.BCEWithLogitsLoss()

        # Optimizers
        self.optimizer_G = torch.optim.Adam(
            self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.optimizer_D = torch.optim.Adam(
            self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    def train(self, num_epochs=10):
        """Train the Pix2Pix3D model with adversarial loss."""
        for epoch in range(num_epochs):
            epoch_loss_G = 0.0
            epoch_loss_D = 0.0

            for i, (images, targets) in enumerate(self.dataloader):
                try:
                    # Move data to device
                    images = images.to(self.device)
                    targets = targets.to(self.device)
                    self.optimizer_D.zero_grad()

                    # Real samples
                    real_preds = self.discriminator(targets)
                    real_loss = self.adversarial_loss(
                        real_preds, torch.ones_like(real_preds))

                    # Fake samples
                    generated_images = self.generator(images).detach()
                    fake_preds = self.discriminator(generated_images)
                    fake_loss = self.adversarial_loss(
                        fake_preds, torch.zeros_like(fake_preds))

                    # Total discriminator loss
                    loss_D = (real_loss + fake_loss) / 2
                    loss_D.backward()
                    self.optimizer_D.step()
                    self.optimizer_G.zero_grad()

                    # Adversarial loss
                    fake_preds = self.discriminator(generated_images)
                    adv_loss = self.adversarial_loss(
                        fake_preds, torch.ones_like(fake_preds))

                    # Reconstruction loss
                    recon_loss = self.l1_loss(generated_images, targets)

                    # Total generator loss
                    loss_G = adv_loss + 100 * recon_loss  # Weighting L1 loss
                    loss_G.backward()
                    self.optimizer_G.step()

                    # Accumulate losses
                    epoch_loss_D += loss_D.item()
                    epoch_loss_G += loss_G.item()

                    # Logging
                    print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(self.dataloader)}], "
                          f"D Loss: {loss_D.item():.4f}, G Loss: {loss_G.item():.4f}")

                except Exception as e:
                    print(f"Error during training: {e}")
                    continue  # Skip the batch if any error occurs

            # Save model checkpoints
            torch.save(self.generator.state_dict(),
                       os.path.join(self.save_dir, "generator.pth"))
            torch.save(self.discriminator.state_dict(),
                       os.path.join(self.save_dir, "discriminator.pth"))
            print(
                f"Epoch {epoch+1} complete. Generator and discriminator saved.")

    def infer(self, input_paths, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        # Handle single directory or list of paths
        if isinstance(input_paths, str) and os.path.isdir(input_paths):
            input_paths = [
                os.path.join(input_paths, f)
                for f in os.listdir(input_paths) if f.endswith(('.jpg', '.png', '.jpeg'))
            ]
        elif not isinstance(input_paths, list):
            raise ValueError(
                "input_paths must be a directory or a list of image paths.")

        self.generator.eval()
        with torch.no_grad():
            for image_path in input_paths:
                try:
                    # Preprocess image
                    combined_input = self.preprocess_image(image_path)
                    # Generate output
                    generated_image = self.generator(
                        combined_input).cpu().squeeze(0)
                    # Save output
                    output_image = transforms.ToPILImage()(generated_image)
                    output_name = os.path.basename(image_path)
                    output_path = os.path.join(
                        output_dir, f"generated_{output_name}")
                    output_image.save(output_path)

                    print(f"Generated: {output_path}")
                except Exception as e:
                    print(f"Failed to process {image_path}: {e}")


# Main Script
if __name__ == "__main__":
    dataset_path = './data/car_images'
    save_dir = './Models/'
    model = Pix2Pix3DModel(title="Car3D", training=True,
                           dataset_path=dataset_path, save_dir=save_dir)
    model.train(num_epochs=5)

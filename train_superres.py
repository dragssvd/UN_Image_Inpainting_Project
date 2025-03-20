import comet_ml
import torch
from torchvision import transforms
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import einops
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from datasets import load_dataset, DatasetDict
from torch.utils.data import DataLoader  # NOQA
import torchvision.transforms as T
from datasets import load_dataset  # NOQA
from torchinfo import summary  # NOQA
from tqdm import tqdm  # NOQA
from comet_ml.integration.pytorch import log_model  # NOQA
from torchmetrics.functional.image import structural_similarity_index_measure  # NOQA
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from SuperresolutionAutoencoder import SuperresolutionAutoencoder
from create_mask import generate_scaled_blob
from torchvision import models
import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import random
from torch import autocast, GradScaler
from PIL import Image


class WikiArtDataset(Dataset):
    def __init__(self, image_files, transform=None):
        self.image_files = image_files
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image


def create_datasets(dataset_path, transforms):
    all_image_files = np.array([os.path.join(dataset_path, f) for f in os.listdir(dataset_path) if f.endswith('.png')])

    # Split dataset indices
    data_size = len(all_image_files)
    indices = list(range(data_size))
    random.shuffle(indices)

    # 80:20 split for training+validation and testing
    test_split = int(0.8 * data_size)
    train_val_indices = indices[:test_split]
    test_indices = indices[test_split:]

    # Further split training+validation into 90:10
    train_split = int(0.9 * len(train_val_indices))
    train_indices = train_val_indices[:train_split]
    val_indices = train_val_indices[train_split:]

    train_data = all_image_files[train_indices]
    val_data = all_image_files[val_indices]
    test_data = all_image_files[test_indices]

    train_dataset = WikiArtDataset(train_data, transform=transforms)
    val_dataset = WikiArtDataset(val_data, transform=transforms)
    test_dataset = WikiArtDataset(test_data, transform=transforms)

    return train_dataset, val_dataset, test_dataset




def setup_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
        # Set default tensor type for cuda
        torch.set_default_dtype(torch.float32)
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
        # Ensure we're using float32 on CPU
        torch.set_default_dtype(torch.float64)
    return device


class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        self.feature_extractor = models.vgg16(pretrained=True).features.eval()
        self.layers = ['4', '9', '16']
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

    def forward(self, predicted, targets):
        pred_features = self.extract_features(predicted)
        target_features = self.extract_features(targets)

        loss = 0.0
        for pred_feat, target_feat in zip(pred_features, target_features):
            loss += nn.functional.mse_loss(pred_feat, target_feat)
        return loss

    def extract_features(self, x):
        features = []
        for name, layer in self.feature_extractor._modules.items():
            x = layer(x)
            if name in self.layers:
                features.append(x)
        return features


class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super(CombinedLoss, self).__init__()
        self.perceptual_loss = PerceptualLoss()
        self.mse_loss = nn.MSELoss()
        self.alpha = alpha

    def forward(self, predicted, targets):
        mse = self.mse_loss(predicted, targets)
        perceptual = self.perceptual_loss(predicted, targets)

        return self.alpha * mse + (1 - self.alpha) * perceptual



if __name__ == '__main__':

    # PARAMETERS TO SET
    COMET_API_KEY = "LP4wJZSrJYL1KJZ06ahrmPLUb"
    PATH = "models/upsample/mse_perceptual/model2.pth"
    BATCH_SIZE = 16
    MAX_EPOCH = 20
    WORKERS = 16
    PIN_MEMORY = True
    LR = 3e-5


    device = setup_device()
    print(f"Using {device} device")

    # obrazy już są 512x512
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    data_path = 'dataset'
    train_dataset, val_dataset, test_dataset = create_datasets(data_path, transform)

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=WORKERS, shuffle=True, pin_memory=PIN_MEMORY)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=WORKERS, shuffle=True, pin_memory=PIN_MEMORY)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=WORKERS,shuffle=True, pin_memory=PIN_MEMORY)


    model = SuperresolutionAutoencoder(input_channels=3)
    print("Model imported")

    # SET UP COMET ML
    comet_experiment = comet_ml.Experiment(
        api_key=COMET_API_KEY,
        project_name="UczenieNienadzorowane")
    comet_experiment.log_code(folder="/UN")
    comet_experiment.log_parameters(
        {
            "batch_size": train_dataloader.batch_size,
        }
    )

    summ = summary(model, (32, 3, 256, 256), device=device, depth=5)
    comet_experiment.set_model_graph(f"{model.__repr__()}\n{summ}")

    num_epochs = MAX_EPOCH
    loss_func = CombinedLoss(alpha=0.7).to(device)#nn.MSELoss(reduction='none')

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    comet_experiment.log_parameter("num_epochs", num_epochs)
    comet_experiment.log_parameter("loss_func", "MSE_LOSS")

    comet_experiment.add_tag(f"LOSS: MSE+PerceptualLoss")

    print("COMET SET")
    print("Start Training")


    # Przygotowanie GradScaler
    scaler = GradScaler()
    # Train and validate
    for epoch in range(num_epochs):
        comet_experiment.set_epoch(epoch)

        # Training phase
        model.train()
        with comet_experiment.train() as train:
            for idx, batch in tqdm(enumerate(train_dataloader), desc=f"TRAIN_{epoch}"):
                comet_experiment.set_step(idx + epoch * len(train_dataloader))

                optimizer.zero_grad()  # Zero out gradients before each batch

                images = F.interpolate(batch, size=(256, 256), mode='bilinear', align_corners=False)
                # nie ma normalizacji bo to_tensor() domyślnie skaluje pil na zakres [0-1]

                inputs = images.to(device)
                targets = batch.to(device)

                with autocast(device_type="cuda"):  # Włączenie mixed precision
                    reconstructed_image = model(inputs)  # Forward pass w FP16 tam, gdzie to możliwe
                    loss = loss_func(reconstructed_image, targets)


                # Calculate SSIM
                metric = structural_similarity_index_measure(reconstructed_image, targets, data_range=1.0)


                # Backpropagation
                # loss.backward()
                # optimizer.step()
                scaler.scale(loss).backward()  # Skalowanie gradientów
                scaler.step(optimizer)         # Aktualizacja wag
                scaler.update()                # Aktualizacja skali dla GradScaler

                # Log metrics to Comet
                comet_experiment.log_metric("loss", loss.item())
                comet_experiment.log_metric("SSIM", metric.item())

                # # Optionally log latents if desired (e.g., logging feature maps or bottleneck representation)
                # if not idx % 50:
                #     # You can log latents from the encoder if desired, assuming you're capturing them during forward pass
                #     # For now, we assume you might want to log `latents` from the encoder, which you can capture by:
                #     # `latents` = model.encoder(images)  # Capture latents from encoder (modify accordingly)
                #     # comet_experiment.log_histogram_3d(latents.detach().cpu(), "latents")
                #     pass

        # Validation phase
        model.eval()
        with comet_experiment.validate() as validat, torch.no_grad() as nograd:
            for idx, batch in tqdm(enumerate(val_dataloader), desc=f"VAL_{epoch}"):
                comet_experiment.set_step(idx + epoch * len(val_dataloader))

                images = F.interpolate(batch, size=(256, 256), mode='bilinear', align_corners=False)
                # nie ma normalizacji bo to_tensor() domyślnie skaluje pil na zakres [0-1]

                inputs = images.to(device)
                targets = batch.to(device)

                with autocast(device_type="cuda"):  # Włączenie mixed precision
                    reconstructed_image = model(inputs)
                    loss = loss_func(reconstructed_image, targets)


                metric = structural_similarity_index_measure(reconstructed_image, targets, data_range=1.0)

                comet_experiment.log_metric("loss", loss.item())
                comet_experiment.log_metric("SSIM", metric.item())



        print("Save model dict")
        torch.save(model.state_dict(), PATH)
        print("Save model")
        torch.save(model, PATH)




    # Set the model to evaluation mode
    model.eval()

    print("Start test")
    # Begin the test phase
    with comet_experiment.test() as test, torch.no_grad():
        for idx, batch in tqdm(enumerate(test_dataloader), desc=f"TEST_{num_epochs}"):
            comet_experiment.set_step(idx + num_epochs * len(test_dataloader))

            # Load and preprocess images
            images = F.interpolate(batch, size=(256, 256), mode='bilinear', align_corners=False)
            # nie ma normalizacji bo to_tensor() domyślnie skaluje pil na zakres [0-1]


            # targets = targets.to(device)
            inputs = images.to(device)
            targets = batch.to(device)

            with autocast(device_type="cuda"):  # Włączenie mixed precision
                reconstructed_image = model(inputs)
                loss = loss_func(reconstructed_image, targets)

            # Compute SSIM metric
            metric = structural_similarity_index_measure(reconstructed_image, targets, data_range=1.0)


            # Log metrics to Comet
            comet_experiment.log_metric("loss", loss.item())
            comet_experiment.log_metric("SSIM", metric.item())

            if idx < 4:
                try:
                    # Convert the images and predictions to HWC format (Height-Width-Channel)
                    images = einops.rearrange(
                        images, "batch channel height width -> batch height width channel"
                    ).cpu().detach().numpy()

                    reconstructed_image = einops.rearrange(
                        reconstructed_image, "batch channel height width -> batch height width channel"
                    ).cpu().detach().numpy()

                    # Clip pixel values to the [0, 1] range
                    images = np.clip(images, 0, 1)
                    reconstructed_image = np.clip(reconstructed_image, 0, 1)

                    # Convert to uint8 for visualization
                    images = (images * 255).astype(np.uint8)
                    reconstructed_image = (reconstructed_image * 255).astype(np.uint8)

                    # Combine images side by side for comparison (original vs. reconstructed)
                    combined_image = np.hstack((images[0], reconstructed_image[0]))  # Side-by-side comparison

                    # Log the combined image
                    comet_experiment.log_image(combined_image, name=f"comparison_{idx}", step=num_epochs)

                except Exception as e:
                    # Error handling
                    print(f"Error during image logging: {e}")
                    print(f"Tensor shape: {images.shape if 'images' in locals() else 'N/A'}")

    print("Save model dict")
    torch.save(model.state_dict(), PATH)
    print("Save model")
    torch.save(model, PATH)


    print("Show results from test")
    for idx, batch in tqdm(enumerate(test_dataloader), desc=f"TEST_{num_epochs}"):
        images = F.interpolate(batch, size=(256, 256), mode='bilinear', align_corners=False)
        # nie ma normalizacji bo to_tensor() domyślnie skaluje pil na zakres [0-1]



        images = images.to(device)

        # Model predictions and latents
        predictions = model(images)

        images = (images * 255).byte()
        predictions = (predictions * 255).byte()

        fig, axes = plt.subplots(nrows=BATCH_SIZE, ncols=2, figsize=(10, 160))  # Adjust figure size as needed
        for i in range(BATCH_SIZE):
            # Original Image
            image_np = np.array(images[i].to('cpu'))
            axes[i, 0].imshow(image_np.transpose(1, 2, 0))
            axes[i, 0].axis('off')  # Turn off axis for cleaner display
            axes[i, 0].set_title(f"Original {i}")

            # Prediction Image
            prediction_np = np.array(predictions[i].to('cpu'))
            axes[i, 1].imshow(prediction_np.transpose(1, 2, 0))
            axes[i, 1].axis('off')
            axes[i, 1].set_title(f"Prediction {i}")

        plt.tight_layout()
        plt.show()

        if idx == 4:
            break # Only process the first four batches


    print("Save model dict")
    torch.save(model.state_dict(), PATH)
    print("Save model")
    torch.save(model, PATH)


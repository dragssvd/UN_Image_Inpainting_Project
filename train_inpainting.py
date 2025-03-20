import comet_ml

import torch
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
from InpaintingAutoencoderWithSkipCons import InpainintgAutoencoderWithSkipCons
from create_mask import generate_scaled_blob
from torchvision import models

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
        self.mse_loss = nn.MSELoss(reduction='none')
        self.alpha = alpha

    def forward(self, predicted, targets, masks):
        loss_weights = torch.where(masks == 0.0, torch.tensor(1.0), torch.tensor(10.0))
        loss_weights = loss_weights.unsqueeze(1)  # batch 1 h w
        loss_weights = loss_weights.to(predicted.device)


        mse = self.mse_loss(predicted, targets) * loss_weights
        mse = mse.mean()

        perceptual = self.perceptual_loss(predicted, targets)

        return self.alpha * mse + (1 - self.alpha) * perceptual



if __name__ == '__main__':

    # PARAMETERS TO SET
    COMET_API_KEY = "LP4wJZSrJYL1KJZ06ahrmPLUb"
    PATH = "imgs/autoenkoder_inpating/mse_perceptual/model1.pth"
    BATCH_SIZE = 32
    MAX_EPOCH = 20
    WORKERS = 16
    PIN_MEMORY = True
    LR = 3e-5


    device = setup_device()
    print(f"Using {device} device")

    ds = load_dataset("Artificio/WikiArt_Full").with_format("torch")

    # 90% train, 10% test + validation
    train_testvalid = ds['train'].train_test_split(test_size=0.2)
    # Split the 10% test + valid in half test, half valid
    test_valid = train_testvalid['test'].train_test_split(test_size=0.5)
    # gather everyone if you want to have a single DatasetDict
    train_test_valid_dataset = DatasetDict({
        'train': train_testvalid['train'],
        'test': test_valid['test'],
        'valid': test_valid['train']}).with_format("torch")

    ds = train_test_valid_dataset
    print(ds)

    train_loader = DataLoader(
        ds["train"],
        batch_size=BATCH_SIZE,
        num_workers=WORKERS,
        shuffle=True,
        pin_memory=PIN_MEMORY,
    )

    test_loader = DataLoader(
        ds["test"],
        batch_size=BATCH_SIZE,
        num_workers=WORKERS,
        shuffle=False,
        pin_memory=PIN_MEMORY,
    )

    val_loader = DataLoader(
        ds["valid"],
        batch_size=BATCH_SIZE,
        num_workers=WORKERS,
        shuffle=False,
        pin_memory=PIN_MEMORY,
    )


    model = InpainintgAutoencoderWithSkipCons(input_channels=4)

    print("Model imported")

    # SET UP COMET ML
    comet_experiment = comet_ml.Experiment(
        api_key=COMET_API_KEY,
        project_name="UczenieNienadzorowane")
    comet_experiment.log_code(folder="/UN")
    comet_experiment.log_parameters(
        {
            "batch_size": train_loader.batch_size,
            "train_size": ds["train"].num_rows,
            "val_size": ds["valid"].num_rows,
        }
    )

    summ = summary(model, (32, 4, 256, 256), device=device, depth=5)
    comet_experiment.set_model_graph(f"{model.__repr__()}\n{summ}")

    num_epochs = MAX_EPOCH
    loss_func = CombinedLoss(alpha=0.7).to(device)#nn.MSELoss(reduction='none')


    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    comet_experiment.log_parameter("num_epochs", num_epochs)
    comet_experiment.log_parameter("loss_func", "MSE_LOSS")

    comet_experiment.add_tag(f"LOSS: MSE+PerceptualLoss")

    print("COMET SET")
    print("Start Training")

    # Train and validate
    for epoch in range(num_epochs):
        comet_experiment.set_epoch(epoch)

        # Validation phase
        model.eval()
        with comet_experiment.validate() as validat, torch.no_grad() as nograd:
            for idx, batch in tqdm(enumerate(val_loader), desc=f"VAL_{epoch}"):
                comet_experiment.set_step(idx + epoch * len(val_loader))

                images = batch["image"].float() / 255.0

                # masks with corrupted part of image
                masks = generate_scaled_blob(images.shape, mask_percentage=(1 / 16) * 100).float() / 255.0

                #apply mask
                images_with_mask = images * (1- masks.unsqueeze(1))

                #add mask as 4th channel
                images_with_mask = torch.cat((images_with_mask, masks.unsqueeze(1)), dim=1) # batch 4 imgsize imgsize


                inputs = images_with_mask.to(device)
                targets = images.to(device)

                # Forward pass through the model
                # Autoencoder directly outputs the reconstructed image
                reconstructed_image = model(inputs)


                # Compute loss between reconstructed image and original image
                # loss_weights = torch.where(masks == 0.0, torch.tensor(1.0), torch.tensor(10.0))
                # loss_weights = loss_weights.unsqueeze(1)  # batch 1 h w
                # loss_weights = loss_weights.to(device)


                # loss = loss_func(reconstructed_image, targets) * loss_weights
                # loss = loss.mean()
                loss = loss_func(reconstructed_image, targets, masks)


                # Calculate SSIM
                metric = structural_similarity_index_measure(reconstructed_image, targets, data_range=1.0)


                # Log metrics to Comet
                comet_experiment.log_metric("loss", loss.item())
                comet_experiment.log_metric("SSIM", metric.item())

                # Optionally log images (showing reconstructed vs original)
                # if idx < 4:
                #     images = einops.rearrange(
                #         [images, reconstructed_image],
                #         "source batch 1 height width -> batch height (source width)",
                #     ).cpu()
                #     comet_experiment.log_image(images[0], f"images_{idx}", step=epoch)

        # Training phase
        model.train()
        with comet_experiment.train() as train:
            for idx, batch in tqdm(enumerate(train_loader), desc=f"TRAIN_{epoch}"):
                comet_experiment.set_step(idx + epoch * len(train_loader))

                optimizer.zero_grad()  # Zero out gradients before each batch

                images = batch["image"].float() / 255.0
                labels = batch["style"]  # Assuming you still have this in your dataset

                # masks with corrupted part of image
                masks = generate_scaled_blob(images.shape, mask_percentage=(1 / 16) * 100).float() / 255.0

                # apply mask
                images_with_mask = images * (1 - masks.unsqueeze(1))

                # add mask as 4th channel
                images_with_mask = torch.cat((images_with_mask, masks.unsqueeze(1)), dim=1)  # batch 4 imgsize imgsize


                # masks for targets
                # origin_image_masks = torch.zeros(masks.shape)
                # targets = torch.cat((images, origin_image_masks.unsqueeze(1)), dim=1)

                #
                # images = images_with_mask.to(device)
                # targets = targets.to(device)

                inputs = images_with_mask.to(device)
                targets = images.to(device)


                # Forward pass through the model
                # Autoencoder directly outputs the reconstructed image
                reconstructed_image = model(inputs)

                # Compute loss between reconstructed image and original image
                # loss_weights = torch.where(masks == 0.0, torch.tensor(1.0), torch.tensor(10.0))
                # loss_weights = loss_weights.unsqueeze(1)  # batch 1 h w
                # loss_weights = loss_weights.to(device)

                #
                # loss = loss_func(reconstructed_image, targets) * loss_weights
                # loss = loss.mean()
                loss = loss_func(reconstructed_image, targets, masks)

                # Calculate SSIM
                metric = structural_similarity_index_measure(reconstructed_image, targets, data_range=1.0)


                # Backpropagation
                loss.backward()
                optimizer.step()



                # Log metrics to Comet
                comet_experiment.log_metric("loss", loss.item())
                comet_experiment.log_metric("SSIM", metric.item())

                # Optionally log latents if desired (e.g., logging feature maps or bottleneck representation)
                if not idx % 50:
                    # You can log latents from the encoder if desired, assuming you're capturing them during forward pass
                    # For now, we assume you might want to log `latents` from the encoder, which you can capture by:
                    # `latents` = model.encoder(images)  # Capture latents from encoder (modify accordingly)
                    # comet_experiment.log_histogram_3d(latents.detach().cpu(), "latents")
                    pass



    # Set the model to evaluation mode
    model.eval()

    print("Start test")
    # Begin the test phase
    with comet_experiment.test() as test, torch.no_grad():
        for idx, batch in tqdm(enumerate(test_loader), desc=f"TEST_{num_epochs}"):
            comet_experiment.set_step(idx + num_epochs * len(test_loader))

            # Load and preprocess images
            images = batch["image"].float() / 255.0  # Assuming images are in [0, 255], normalize to [0, 1]
            labels = batch["style"]  # Assuming you still have this in your dataset (though not used here)

            # masks with corrupted part of image
            masks = generate_scaled_blob(images.shape, mask_percentage=(1 / 16) * 100).float() / 255.0

            # apply mask
            images_with_mask = images * (1 - masks.unsqueeze(1))

            # add mask as 4th channel
            images_with_mask = torch.cat((images_with_mask, masks.unsqueeze(1)), dim=1)  # batch 4 imgsize imgsize

            # masks for targets
            # origin_image_masks = torch.zeros(masks.shape)
            # targets = torch.cat((images, origin_image_masks.unsqueeze(1)), dim=1)

            #
            # images = images_with_mask.to(device)
            # targets = targets.to(device)
            inputs = images_with_mask.to(device)
            targets = images.to(device)


            # Forward pass through the model
            # Autoencoder directly outputs the reconstructed image
            reconstructed_image = model(inputs)

            # Compute loss between reconstructed image and original image
            # loss_weights = torch.where(masks == 0.0, torch.tensor(1.0), torch.tensor(10.0))
            # loss_weights = loss_weights.unsqueeze(1)  # batch 1 h w
            # loss_weights = loss_weights.to(device)
            #
            # loss = loss_func(reconstructed_image, targets) * loss_weights
            # loss = loss.mean()
            loss = loss_func(reconstructed_image, targets, masks)


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
    for idx, batch in tqdm(enumerate(test_loader), desc=f"TEST_{num_epochs}"):
        images = batch["image"].float() / 255.0
        labels = batch["style"]

        # masks with corrupted part of image
        masks = generate_scaled_blob(images.shape, mask_percentage=(1 / 16) * 100).float() / 255.0

        # apply mask
        images_with_mask = images * (1 - masks.unsqueeze(1))

        # add mask as 4th channel
        images_with_mask = torch.cat((images_with_mask, masks.unsqueeze(1)), dim=1)  # batch 4 imgsize imgsize

        images = images_with_mask.to(device)

        # Model predictions and latents
        predictions = model(images)

        images = (images * 255).byte()
        predictions = (predictions * 255).byte()

        fig, axes = plt.subplots(nrows=32, ncols=2, figsize=(10, 160))  # Adjust figure size as needed
        for i in range(32):
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


import comet_ml
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader  # NOQA
from datasets import load_dataset  # NOQA
from torchinfo import summary  # NOQA
from tqdm import tqdm  # NOQA
from comet_ml.integration.pytorch import log_model  # NOQA
from torchmetrics.functional.image import structural_similarity_index_measure  # NOQA
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from upscaling_treining import create_datasets
from torchvision import transforms
from torch.utils.data import Dataset
import random
from PIL import Image
import os


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

if __name__ == '__main__':
    BATCH_SIZE = 16
    MAX_EPOCH = 20
    WORKERS = 16
    PIN_MEMORY = True
    LR = 3e-5









    device = setup_device()

    print(f"Using {device} device")

    PATH = "models/upsample/mse_perceptual/model2.pth"
    # model = VGG16Autoencoder()
    new_model = torch.load(PATH)
    new_model.eval()
    new_model.to(device)
    print(new_model.state_dict())

    # new_model = VGG16Autoencoder(input_channels=4)
    # new_model.load_state_dict(model.state_dict())
    # new_model.eval()
    # new_model.to(device)
    # model.load_state_dict(m.state_dict())
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    data_path = 'dataset'
    train_dataset, val_dataset, test_dataset = create_datasets(data_path, transform)

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=WORKERS, shuffle=True,
                                  pin_memory=PIN_MEMORY)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=WORKERS, shuffle=True,
                                pin_memory=PIN_MEMORY)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=WORKERS, shuffle=True,
                                 pin_memory=PIN_MEMORY)

    print("Show results from test")
    for idx, batch in tqdm(enumerate(test_dataloader), desc=f"TEST_{0}"):
        images = F.interpolate(batch, size=(256, 256), mode='bilinear', align_corners=False)
       # images = images.float() / 255.0
        images = images.cuda()

        # Model predictions and latents
        predictions = new_model(images)

        images = (images * 255).byte()
        predictions = (predictions * 255).byte()
        origin_data = (batch * 255).byte()


        fig, axes = plt.subplots(nrows=BATCH_SIZE, ncols=5, figsize=(30, 160))  # Adjust figure size as needed
        print(images.shape)
        print(predictions.shape)
        for i in range(BATCH_SIZE):


            # Original Image
            image_np = np.array(images[i].to('cpu'))
            axes[i, 0].imshow(image_np.transpose(1, 2, 0))
            axes[i, 0].axis('off')  # Turn off axis for cleaner display
            axes[i, 0].set_title(f"Input 256x256 {i}")


            # Prediction Image
            prediction_np = np.array(predictions[i].to('cpu'))
            axes[i, 1].imshow(prediction_np.transpose(1, 2, 0))
            axes[i, 1].axis('off')
            axes[i, 1].set_title(f"Prediction 512x512 {i}")

            # Original Image
            origin_np = np.array(origin_data[i].to('cpu'))
            axes[i, 2].imshow(origin_np.transpose(1, 2, 0))
            axes[i, 2].axis('off')  # Turn off axis for cleaner display
            axes[i, 2].set_title(f"Original 512x512 {i}")


            diff = np.array(predictions[i].to('cpu') - batch[i].to('cpu'))
            result_clipped = np.clip(diff, 0, 255).astype(np.uint8)
            axes[i, 3].imshow(result_clipped.transpose(1, 2, 0))
            axes[i, 3].axis('off')
            axes[i, 3].set_title(f"prediction-original {i}")



        plt.tight_layout()
        plt.show()

        if idx == 4:
            break  # Only process the first four batches



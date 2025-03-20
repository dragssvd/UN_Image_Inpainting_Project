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
from torchmetrics.functional.image import structural_similarity_index_measure  # NOQA
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from InpaintingAutoencoderWithSkipCons import InpainintgAutoencoderWithSkipCons
from create_mask import generate_scaled_blob


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
    BATCH_SIZE = 32
    MAX_EPOCH = 20
    WORKERS = 16
    PIN_MEMORY = True
    LR = 3e-5









    device = setup_device()

    print(f"Using {device} device")

    gim_path = "models/submodels/cluster_0_submodel.pth"
    new_model = torch.load(gim_path, map_location=device)
    new_model.to(device)
    new_model.eval()
    print(new_model.state_dict())

    # new_model = VGG16Autoencoder(input_channels=4)
    # new_model.load_state_dict(model.state_dict())
    # new_model.eval()
    # new_model.to(device)
    # model.load_state_dict(m.state_dict())

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


    test_loader = DataLoader(
        ds["test"],
        batch_size=BATCH_SIZE,
        num_workers=WORKERS,
        shuffle=False,
        pin_memory=PIN_MEMORY,
    )









    print("Show results from test")
    for idx, batch in tqdm(enumerate(test_loader), desc=f"TEST_{0}"):
        images = batch["image"].float() / 255.0
        labels = batch["style"]


        masks = generate_scaled_blob(images.shape, mask_percentage=(1 / 16) * 100).float() / 255.0

        images_with_mask = images * (1 - masks.unsqueeze(1))
        images_with_mask = torch.cat((images_with_mask, masks.unsqueeze(1)), dim=1)  # batch 4 imgsize imgsize

        images = images_with_mask.to(device)
        # Model predictions and latents
        predictions = new_model(images)

        images = (images * 255).byte()
        predictions = (predictions * 255).byte()





        fig, axes = plt.subplots(nrows=32, ncols=2, figsize=(10, 160))  # Adjust figure size as needed
        for i in range(32):
            # Original Image
            image_np = np.array(images[i].to('cpu'))
            axes[i, 0].imshow(image_np.transpose(1, 2, 0)[:,:,:3])
            axes[i, 0].axis('off')  # Turn off axis for cleaner display
            axes[i, 0].set_title(f"Original {i}")

            # Prediction Image
            prediction_np = np.array(predictions[i].to('cpu'))
            axes[i, 1].imshow(prediction_np.transpose(1, 2, 0)[:,:,:3])
            axes[i, 1].axis('off')
            axes[i, 1].set_title(f"Prediction {i}")

        plt.tight_layout()
        plt.show()

        if idx == 4:
            break # Only process the first four batches

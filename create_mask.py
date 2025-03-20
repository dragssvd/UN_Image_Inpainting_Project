

import torch
import numpy as np
import random
import cv2
from scipy.interpolate import splprep, splev


def generate_scaled_blob_torch(image_shape, mask_percentage):


    """
    Generuje nieregularną, gładką plamę skalowaną do zadanego procentu powierzchni obrazu.

    :param image_shape: Kształt obrazu w formacie (batch_size, height, width, channels).
    :param mask_percentage: Procent powierzchni obrazu pokryty plamą.
    :return: Maska (Tensor PyTorch) z wygładzoną nieregularną plamą.
    """
    batch_size, _, height, width = image_shape

    # Oblicz docelową liczbę pikseli dla plamy
    total_pixels = height * width
    target_pixels = total_pixels * (mask_percentage / 100)

    # Stworzenie tensorów do przechowywania masek
    masks = torch.zeros(batch_size, height, width)

    for i in range(batch_size):
        # Losowy punkt startowy plamy
        center_x = random.randint(width // 4, 3 * width // 4)
        center_y = random.randint(height // 4, 3 * height // 4)

        # Generacja nieregularnego konturu plamy
        num_points = random.randint(8, 15)  # Liczba punktów definiujących plamę
        angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
        radii = np.random.uniform(low=0.1, high=0.3, size=num_points) * min(height, width)
        points = np.array([(
                center_x + r * np.cos(angle),
                center_y + r * np.sin(angle)
            ) for angle, r in zip(angles, radii)], dtype=np.float32)

        # Zamknięcie konturu (dodanie pierwszego punktu na koniec)
        points = np.vstack([points, points[0]])

        # Interpolacja krzywą splajnu, aby wygładzić kontur
        tck, u = splprep([points[:, 0], points[:, 1]], s=0.5, per=True)
        u_new = np.linspace(0, 1, 100)  # Więcej punktów dla płynności
        x_new, y_new = splev(u_new, tck)

        # Konwersja na współrzędne całkowite
        smooth_points = np.array([x_new, y_new]).T.astype(np.int32)

        # Stwórz pustą maskę i narysuj kontur
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask, [smooth_points], 255)

        # Oblicz aktualny rozmiar plamy
        current_pixels = np.sum(mask > 0)

        # Oblicz współczynnik skalowania
        scale_factor = (target_pixels / current_pixels) ** 0.5 if current_pixels > 0 else 1.0

        # Skalowanie maski
        if scale_factor != 1.0:
            # Przeskaluj maskę do odpowiedniego rozmiaru
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            scaled_mask = cv2.resize(mask, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

            # Środek oryginalnego obrazu
            final_mask = np.zeros_like(mask)
            y_offset = (height - new_height) // 2
            x_offset = (width - new_width) // 2
            final_mask[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = scaled_mask
        else:
            final_mask = mask

        # Przekształć maskę do tensora PyTorch
        masks[i] = torch.tensor(final_mask, dtype=torch.float32)

    return masks


def generate_scaled_blob(images, mask_percentage):

    """
    Generuje nieregularną, gładką plamę skalowaną do zadanego procentu powierzchni obrazu.

    :param image_shape: Kształt obrazu w formacie (batch_size, height, width, channels).
    :param mask_percentage: Procent powierzchni obrazu pokryty plamą.
    :return: Maska (Tensor PyTorch) z wygładzoną nieregularną plamą.
    """
    batch_size, _, height, width = images

    # Oblicz docelową liczbę pikseli dla plamy
    total_pixels = height * width
    target_pixels = total_pixels * (mask_percentage / 100)

    # Stworzenie tensorów do przechowywania masek
    masks = torch.zeros(batch_size, height, width)

    for i in range(batch_size):

        center_x = random.randint(width // 4, 3 * width // 4)
        center_y = random.randint(height // 4, 3 * height // 4)

        # Generacja nieregularnego konturu plamy
        num_points = random.randint(8, 15)  # Liczba punktów definiujących plamę
        angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
        radii = np.random.uniform(low=0.1, high=0.3, size=num_points) * min(height, width)
        points = np.array([(
                center_x + r * np.cos(angle),
                center_y + r * np.sin(angle)
            ) for angle, r in zip(angles, radii)], dtype=np.float32)

        # Zamknięcie konturu (dodanie pierwszego punktu na koniec)
        points = np.vstack([points, points[0]])

        # Interpolacja krzywą splajnu, aby wygładzić kontur
        tck, u = splprep([points[:, 0], points[:, 1]], s=0.5, per=True)
        u_new = np.linspace(0, 1, 100)  # Więcej punktów dla płynności
        x_new, y_new = splev(u_new, tck)

        # Konwersja na współrzędne całkowite
        smooth_points = np.array([x_new, y_new]).T.astype(np.int32)

        # Stwórz pustą maskę i narysuj kontur
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask, [smooth_points], 255)

        # Oblicz aktualny rozmiar plamy
        current_pixels = np.sum(mask > 0)

        # Oblicz współczynnik skalowania
        scale_factor =  (target_pixels / current_pixels) ** 0.5 if current_pixels > 0 else 1.0

        # Skalowanie maski
        if scale_factor <= 1.0:
            scale_factor += -0.3

            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            scaled_mask = cv2.resize(mask, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

            final_mask = np.zeros_like(mask)

            y_offset = (height - new_height)
            x_offset = (width - new_width)

            try:
                x_pos = random.randint(0, x_offset)
                y_pos = random.randint(0, y_offset)
                final_mask[y_pos:y_pos + new_height, x_pos:x_pos + new_width] = scaled_mask
            except:
                print(x_offset)
                print(y_offset)
                print(scale_factor)

        else:
            final_mask = mask
        # Przekształć maskę do tensora PyTorch
        masks[i] = torch.tensor(final_mask, dtype=torch.float32)
    return masks

if __name__ == '__main__':



    import torch
    from torch import nn
    from torch.utils.data import DataLoader
    import numpy as np
    from datasets import load_dataset, DatasetDict
    from torch.utils.data import DataLoader  # NOQA
    import torchvision.transforms as T
    from datasets import load_dataset  # NOQA
    from torchinfo import summary  # NOQA
    from tqdm import tqdm  # NOQA
    from comet_ml.integration.pytorch import log_model  # NOQA
    from torchmetrics.functional.image import structural_similarity_index_measure  # NOQA
    import numpy as np
    from tqdm import tqdm
    from InpaintingAutoencoderWithSkipCons import InpainintgAutoencoderWithSkipCons



    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    BATCH_SIZE = 48
    MAX_EPOCH = 20
    WORKERS = 16
    PIN_MEMORY = True
    LR = 3e-5

    model = InpainintgAutoencoderWithSkipCons(input_channels=4).to(device)
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
    loss_func = nn.MSELoss(reduction='none')
    for idx, batch in tqdm(enumerate(train_loader), desc=f"VAL_{0}"):
        # images = batch["image"].float() / 255.0
        # masks = generate_scaled_blob(images.shape, mask_percentage=(1 / 16) * 100).float() / 255.0
        # images_with_mask = torch.cat((images, masks.unsqueeze(1)), dim=1)
        #
        import einops
        # img = (images[0].numpy() *255).astype(np.uint8)
        # img = einops.rearrange(
        #     img, "channel height width -> height width channel"
        # )

        images = batch["image"].float() / 255.0
        masks = generate_scaled_blob(images.shape, mask_percentage=(1 / 16) * 100).float() / 255.0

        images_with_mask = images * (1 - masks.unsqueeze(1))
        images_with_mask = torch.cat((images_with_mask, masks.unsqueeze(1)), dim=1)  # batch 4 imgsize imgsize



        # masks for targets
        origin_image_masks = torch.zeros(masks.shape)
        targets = torch.cat((images, origin_image_masks.unsqueeze(1)), dim=1)


        img = (images[0].numpy() *255).astype(np.uint8)
        img = einops.rearrange(
            img, "channel height width -> height width channel"
        )
        cv2.imshow("Original Image", img )
        cv2.imshow("Mask", masks[0, :, :].numpy().astype(np.uint8) * 255)




        img_with_mask =  (images_with_mask[0].numpy() * 255).astype(np.uint8)
        img_with_mask = einops.rearrange(
            img_with_mask, "channel height width -> height width channel"
        )

        cv2.imshow("Masked Image", img_with_mask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        break







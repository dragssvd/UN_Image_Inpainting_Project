import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
import io
import os
from create_mask import generate_scaled_blob
from classifier import AdaptedStyleClusterCNN

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

device = setup_device()

print(f"Using {device} device")

gim_array = [
    "models/inpating/submodels/cluster_0_submodel.pth",
    "models/inpating/submodels/cluster_1_submodel.pth",
    "models/inpating/submodels/cluster_2_submodel.pth",
    "models/inpating/submodels/cluster_3_submodel.pth",
    "models/inpating/submodels/cluster_4_submodel.pth",
    "models/inpating/submodels/cluster_5_submodel.pth",
    "models/inpating/submodels/cluster_6_submodel.pth",
    "models/inpating/submodels/cluster_7_submodel.pth",
    "models/inpating/submodels/cluster_8_submodel.pth",
    "models/inpating/submodels/cluster_9_submodel.pth",
    "models/inpating/submodels/cluster_10_submodel.pth",
    "models/inpating/submodels/cluster_11_submodel.pth",
    "models/inpating/submodels/cluster_12_submodel.pth",
    "models/inpating/submodels/cluster_13_submodel.pth",
    "models/inpating/submodels/cluster_14_submodel.pth",
    "models/inpating/submodels/cluster_15_submodel.pth",
    "models/inpating/submodels/cluster_16_submodel.pth",
    "models/inpating/submodels/cluster_17_submodel.pth",
    "models/inpating/submodels/cluster_18_submodel.pth",
    "models/inpating/submodels/cluster_19_submodel.pth"
]

classifier_model = AdaptedStyleClusterCNN(20)

ccm_path = "models/model1_100"
classifier_model.load_state_dict(torch.load(ccm_path, map_location=device))
classifier_model.to(device)
classifier_model.eval()

srm_path = "models/upsample/mse_perceptual/model2.pth"
super_resolution_model = torch.load(srm_path, map_location=device)
super_resolution_model.to(device)
super_resolution_model.eval()

PATH = "models/inpating/mse_perceptual/model1.pth"
# model = VGG16Autoencoder()
new_model = torch.load(PATH)
new_model.eval()
new_model.to(device)

encoder = new_model.encoder
encoder.eval()  # Set the encoder to evaluation mode
encoder.to(device)  # Move the encoder to the appropriate device

# Image processing function
def inpaint_image(image):
        
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    
    images = transform(image).unsqueeze(0)

    #images = image.float() / 255.0

    # masks with corrupted part of image
    masks = generate_scaled_blob(images.shape, mask_percentage=(1 / 16) * 100).float() / 255.0

    #apply mask
    images_with_mask = images * (1- masks.unsqueeze(1))

    #add mask as 4th channel
    images_with_mask = torch.cat((images_with_mask, masks.unsqueeze(1)), dim=1) # batch 4 imgsize imgsize


    preprocessed_image = images_with_mask.to(device)

    original_image_with_blob = preprocessed_image
    
    original_image_with_blob = (original_image_with_blob * 255).byte()  # Remove batch dimension and scale to [0, 255]
    original_image_with_blob = original_image_with_blob.squeeze(0)
    original_image_with_blob = original_image_with_blob.cpu().numpy()  # Convert to a NumPy array on the CPU
    original_image_with_blob = original_image_with_blob.transpose(1, 2, 0)[:,:,:3]  # Rearrange from (C, H, W) to (H, W, C)

    # Convert the NumPy array back to a PIL image
    original_image_with_blob = Image.fromarray(original_image_with_blob)

    with torch.no_grad():  
        latents, _ = encoder(preprocessed_image)
        cluster = classifier_model(latents)
        
    print(cluster)
    cluster_idx = torch.argmax(cluster)
    print(cluster_idx)
    
    PATH = gim_array[cluster_idx]
    new_model = torch.load(PATH, map_location=device)
    new_model.to(device)
    new_model.eval()

        
    with torch.no_grad():
        output = new_model(preprocessed_image)


    # Post-process the model output
    output = (output * 255).byte()  # Remove batch dimension and scale to [0, 255]
    output = output.squeeze(0)
    image_np = output.cpu().numpy()  # Convert to a NumPy array on the CPU
    image_np = image_np.transpose(1, 2, 0)  # Rearrange from (C, H, W) to (H, W, C)

    # Convert the NumPy array back to a PIL image
    output_image = Image.fromarray(image_np)

    return original_image_with_blob, output_image

def upscale_image(image, model):
    
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    
    images = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(images)

    # Post-process the model output
    output = (output * 255).byte()  # Remove batch dimension and scale to [0, 255]
    output = output.squeeze(0)
    image_np = output.cpu().numpy()  # Convert to a NumPy array on the CPU
    image_np = image_np.transpose(1, 2, 0)  # Rearrange from (C, H, W) to (H, W, C)

    # Convert the NumPy array back to a PIL image
    output_image = Image.fromarray(image_np)

    return output_image

# Streamlit app
st.title("Image Inpainting and Superresolution App")
st.write("Upload an image to see the original and processed versions side by side.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    input_image = Image.open(uploaded_file).convert("RGB")

    # Process the image with the first model
    og_image, processed_image = inpaint_image(input_image)

    # Process the image with the second model
    final_image = upscale_image(processed_image, super_resolution_model)

    # Display images side by side
    col1, col2, col3 = st.columns(3)

    with col1:
        st.image(og_image, caption="Original Image", use_container_width=True)

    with col2:
        st.image(processed_image, caption="Inpainted Image", use_container_width=True)

    with col3:
        st.image(final_image, caption="Upscaled Image", use_container_width=True)
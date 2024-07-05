from colorizers import *
import matplotlib.pyplot as plt
import torch.optim as optim
import io
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

from colorizers.ColorizationDataset import ColorizationDataset


class Colorizer:
    def __init__(self, greyscale_image):
        self.greyscale_image = greyscale_image

    def colorize(self):
        # load colorizers
        colorizer_eccv16 = eccv16(pretrained=True).eval()


        # default size to process images is 256x256
        # grab L channel in both original ("orig") and resized ("rs") resolutions
        img = load_img(self.greyscale_image)
        (tens_l_orig, tens_l_rs) = preprocess_img(img, HW=(256, 256))
        out_img_eccv16 = postprocess_tens(tens_l_orig, colorizer_eccv16(tens_l_rs).cpu())

        # Convert the processed tensor to a PIL image and generate base64 String
        color_image = self.generateBase64String(out_img_eccv16)

        return color_image

    def train_model(self):
        # load colorizer
        colorizer_eccv16 = eccv16(pretrained=True).eval()

        # Freeze all parameters initially
        for param in colorizer_eccv16.parameters():
            param.requires_grad = False

        # Unfreeze specific layers of ECCV16 for fine-tuning
        for param in colorizer_eccv16.model8.parameters():
            param.requires_grad = True


        # Define the Mean Squared Error (MSE) loss function
        criterion = nn.MSELoss()

        # Set up optimizers for both models, but only for the parameters that require gradients
        optimizer_eccv16 = optim.Adam(filter(lambda p: p.requires_grad, colorizer_eccv16.parameters()), lr=0.001)

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((256, 256)),
        ])

        dataset = ColorizationDataset('./image_store/train', transform=transform)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

        # Number of epochs to train the model
        num_epochs = 10
        for epoch in range(num_epochs):
            # Set both models to training mode
            colorizer_eccv16.train()

            # Iterate over batches of data
            for L, ab in dataloader:
                L = L.unsqueeze(1)  # Add channel dimension to L channel (NxHxW -> Nx1xHxW)
                ab = ab.permute(0, 3, 1, 2)  # Change dimension order of ab channels (NxHxWx2 -> Nx2xHxW)

                # Fine-tuning ECCV16 model
                #optimizer_eccv16.zero_grad()  # Clear gradients for ECCV16 optimizer
                output_ab_eccv16 = colorizer_eccv16(L)  # Forward pass through ECCV16
                loss_eccv16 = criterion(output_ab_eccv16, ab)  # Compute loss for ECCV16
                loss_eccv16.backward()  # Backward pass to compute gradients
                optimizer_eccv16.step()  # Update ECCV16 parameters

            # Print the loss for both models at the end of each epoch
            print(
                f'Epoch [{epoch + 1}/{num_epochs}], Loss ECCV16: {loss_eccv16.item():.4f}')

        # Save the fine-tuned models' weights
        torch.save(colorizer_eccv16.state_dict(), 'fine_tuned_colorizer_eccv16.pth')

    def generateBase64String(self, img_rgb):
        # Convert to PIL Image
        img_pil = Image.fromarray((img_rgb * 255).astype(np.uint8))

        # Save image to a bytes buffer in JPEG format
        buffer = io.BytesIO()
        img_pil.save(buffer, format="JPEG")  # Save as JPEG
        img_str = base64.b64encode(buffer.getvalue()).decode()

        # Add prefix to JPEG base64 string
        img_str = "data:image/jpeg;base64," + img_str

        return img_str


from torch.utils.data import Dataset
import os
from skimage import io, color
from PIL import Image

class ColorizationDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_name).convert('RGB')
        if self.transform:
            image = self.transform(image)
            # Permute to (H, W, C)
            image = image.permute(1, 2, 0)
            print("Image size (H, W, C):", image.shape)

            # Convert to NumPy array
            image = image.numpy()

            # Ensure the image values are in the range [0, 1]
            if image.max() > 1.0:
                image = image / 255.0
            # Convert to grayscale (L) and color channels (ab)
        # Convert the image to LAB color space
        image_lab = color.rgb2lab(image)
        # Extract the L channel (lightness)
        tens_l = image_lab[:, :, 0]
        # Extract the AB channels (color channels)
        tens_ab = image_lab[:, :, 1:]
        return tens_l,tens_ab

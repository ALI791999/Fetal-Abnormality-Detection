import torch
import glob
import numpy as np
from PIL import Image
import configs


class HC18Data(torch.utils.data.Dataset):
    def __init__(self, data_type, transform=None):
        """
        Initializes the dataset based on the data_type parameter: 'train', 'validation', or 'test'.
        :param data_type: string, either 'train', 'validation', or 'test'
        :param transform: any data transformation to be applied
        """
        self.data_type = data_type
        self.transform = transform

        # Load the datasets based on the data_type parameter
        if self.data_type == 'train':
            self.x_data = sorted(glob.glob(configs.MAIN_DIR + 'training_set/*HC.png'))
            self.y_data = sorted(glob.glob(configs.MAIN_DIR + 'training_set/*_Mask.png'))
            # Debugging: Check if the files were loaded
            if len(self.x_data) == 0 or len(self.y_data) == 0:
                raise ValueError("No training data found. Check the file paths and naming conventions.")
            print(f"Found {len(self.x_data)} training HC images and {len(self.y_data)} mask images.")

        elif self.data_type == 'validation':
            self.x_data = sorted(glob.glob(configs.MAIN_DIR + 'validation_set/*HC.png'))
            self.y_data = sorted(glob.glob(configs.MAIN_DIR + 'validation_set/*_Mask.png'))
            # Debugging: Check if the files were loaded
            if len(self.x_data) == 0 or len(self.y_data) == 0:
                raise ValueError("No validation data found. Check the file paths and naming conventions.")
            print(f"Found {len(self.x_data)} validation HC images and {len(self.y_data)} mask images.")

        elif self.data_type == 'test':
            self.x_data = sorted(glob.glob(configs.MAIN_DIR + 'test_set/*.png'))
            # Debugging: Check if the files were loaded
            if len(self.x_data) == 0:
                raise ValueError("No test data found. Check the file paths and naming conventions.")
            print(f"Found {len(self.x_data)} test images.")
            self.y_data = None  # No mask for the test set

        else:
            raise ValueError("Invalid data type. Choose from 'train', 'validation', or 'test'.")

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        S = (configs.IMAGE_SIZE, configs.IMAGE_SIZE)
        S1 = (configs.IMAGE_RESIZE_PRE_LEARANBLE_RESIZER, configs.IMAGE_RESIZE_PRE_LEARANBLE_RESIZER)

        # image -> [1,H,W] float32 in [0,1], bilinear resize
        x = Image.open(self.x_data[idx]).convert("L").resize(S1, Image.Resampling.BILINEAR)
        x = np.asarray(x, dtype=np.float32) / 255.0
        x = torch.from_numpy(x).unsqueeze(0)  # [1,H,W]

        if self.data_type == 'test':
            return x

        # mask -> [H,W] int64 {0,1}, NEAREST resize (no channel dim)
        y = Image.open(self.y_data[idx]).convert("L").resize(S, Image.Resampling.NEAREST)
        y = np.asarray(y, dtype=np.uint8)
        y = torch.from_numpy((y > 0).astype(np.int64))  # [H,W]

        return x, y
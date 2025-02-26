import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset

class MNISTDataset(Dataset):
    def __init__(self, image_size=28):
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
        ])
        
        self.dataset = datasets.MNIST(
            root='./data',
            train=True,
            download=True,
            transform=self.transform
        )
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        img, _ = self.dataset[idx]  # Ignore labels
        return img

from rectified_flow_pytorch import RectifiedFlow, Unet, Trainer

def main():
    image_size = 28
    mnist_dataset = MNISTDataset(image_size=image_size)
    
    # Simple model for MNIST
    model = Unet(
        dim=32,
        channels=1,  # Grayscale images
        dim_mults=(1, 2, 4),  # Smaller multipliers for low-res images
    )
    model = torch.compile(model, backend="inductor", mode="default", fullgraph=True, dynamic=False)

    rectified_flow = RectifiedFlow(model)

    trainer = Trainer(
        rectified_flow,
        dataset=mnist_dataset,
        batch_size=512,  # Can use larger batches for smaller images
        num_train_steps=30_000,
        results_folder='./results/mnist',
        checkpoints_folder='./checkpoints/mnist',
        accelerate_kwargs={'log_with':'comet_ml'}
    )

    trainer()

if __name__ == "__main__":
    main()
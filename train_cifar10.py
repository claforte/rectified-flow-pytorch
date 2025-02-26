import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset

class CIFAR10Dataset(Dataset):
    def __init__(self, image_size=32):
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
        ])
        
        self.dataset = datasets.CIFAR10(
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
    image_size = 32
    cifar_dataset = CIFAR10Dataset(image_size=image_size)
    
    # Smaller model for CIFAR-10
    model = Unet(
        dim=32,
        channels=3,  # RGB images
        dim_mults=(1, 2, 4),  # Smaller multipliers for low-res images
    )
    model = torch.compile(model, backend="inductor", mode="default", fullgraph=True, dynamic=False)

    rectified_flow = RectifiedFlow(model)

    trainer = Trainer(
        rectified_flow,
        dataset=cifar_dataset,
        batch_size=128,  # Can use larger batches for smaller images
        num_train_steps=50_000,
        results_folder='./results/cifar10',
        checkpoints_folder='./checkpoints/cifar10',
        accelerate_kwargs={'log_with':'comet_ml'},
        calculate_fid=True,
        fid_every=100
    )

    trainer()

if __name__ == "__main__":
    main()
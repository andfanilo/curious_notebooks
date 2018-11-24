import torch
import torchvision
import torchvision.datasets as datasets


def load_mnist(path):
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    train_dataset = datasets.MNIST(
        root=path, train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        root=path, train=False, download=True, transform=transform
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=64, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1000, shuffle=True
    )

    return (train_loader, test_loader)

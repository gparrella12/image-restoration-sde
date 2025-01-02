import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, DistributedSampler, Dataset
from torch.nn.parallel import DistributedDataParallel as DDP

# Custom Dataset with Random Data
class RandomDataset(Dataset):
    def __init__(self, num_samples, input_dim, num_classes):
        self.num_samples = num_samples
        self.input_dim = input_dim
        self.num_classes = num_classes

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Generate random input and target
        x = torch.randn(self.input_dim)
        y = torch.randint(0, self.num_classes, (1,)).item()
        return x, y

def main():
    # Initialize environment variables
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    print(f"[INFO] RANK={rank}, LOCAL_RANK={local_rank}, WORLD_SIZE={world_size}")

    # Initialize process group
    dist.init_process_group(backend="nccl", init_method="env://")

    # Set GPU for this process
    torch.cuda.set_device(local_rank)

    # Parameters for random data
    num_samples = 10000
    input_dim = 28 * 28  # Example input dimension
    num_classes = 10  # Number of output classes
    batch_size = 64

    # Create random dataset and DataLoader
    dataset = RandomDataset(num_samples=num_samples, input_dim=input_dim, num_classes=num_classes)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=2)

    # Define a simple model
    class SimpleNet(nn.Module):
        def __init__(self):
            super(SimpleNet, self).__init__()
            self.fc1 = nn.Linear(input_dim, 128)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(128, num_classes)

        def forward(self, x):
            x = self.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    # Initialize model, loss, and optimizer
    model = SimpleNet().cuda(local_rank)
    model = DDP(model, device_ids=[local_rank])
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    epochs = 5
    for epoch in range(epochs):
        sampler.set_epoch(epoch)  # Shuffle data for each epoch
        model.train()
        epoch_loss = 0.0

        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.cuda(local_rank), targets.cuda(local_rank)

            # Forward and backward
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            if batch_idx % 10 == 0:
                print(f"[INFO] Rank {rank}, Epoch [{epoch+1}/{epochs}], Batch [{batch_idx}/{len(dataloader)}], Loss: {loss.item():.4f}")

        print(f"[INFO] Rank {rank}, Epoch [{epoch+1}/{epochs}] Average Loss: {epoch_loss/len(dataloader):.4f}")

    # Cleanup
    
    print(f"[INFO] Rank {rank} training completed successfully.")
    exit(0)
    
if __name__ == "__main__":
    main()
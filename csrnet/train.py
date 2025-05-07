import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import CSRNet
from dataset import ShanghaiTechDataset, train_transform

import argparse

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CSRNet().to(device)
    dataset = ShanghaiTechDataset(args.data_root, part=args.part, train=True, transform=train_transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    criterion = nn.MSELoss(reduction='sum')
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    best_loss = float('inf')
    os.makedirs(args.save_dir, exist_ok=True)
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        for i, (images, targets) in enumerate(dataloader):
            images = images.to(device)
            targets = targets.to(device)
            outputs = model(images)
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{args.epochs}], Loss: {avg_loss:.4f}")
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), os.path.join(args.save_dir, f'csrnet_best_{args.part}.pth'))
            print("Best model saved.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True, help='Path to ShanghaiTech dataset root')
    parser.add_argument('--part', type=str, default='A', choices=['A', 'B'], help='Dataset part (A or B)')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    args = parser.parse_args()
    train(args) 
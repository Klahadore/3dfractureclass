import torch 
from torch.nn import functional as F
from torch import optim
from torch import nn
from torch.nn import Dropout, Conv3d
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler

from dataset import *
from GroupConv3d import GroupConv3d
from GroupConvTranspose3d import GroupConvTranspose3d
from compatible_max_pool import MaxPool3DWrapper

class Down(nn.Module):
    def __init__(self, in_channels, out_channels, order='middle'):
        super().__init__()
        self.order = order
        if self.order == 'first':
            self.C1 = GroupConv3d(in_channels, out_channels, order='first')
        else:
            self.C1 = GroupConv3d(in_channels, out_channels, order='middle')
        
        self.relu = nn.ReLU(inplace=True) 

        self.dropout = Dropout(p=0.1)
        self.C2 = GroupConv3d(out_channels, out_channels)
        self.pool = MaxPool3DWrapper(kernel_size=(2,2,2), stride=1)
    
    def forward(self, x):
        x = self.C1(x)
        print(x.dtype)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.C2(x)
        x = self.relu(x)
        skip_con = x.clone()
        x = self.pool(x)
        return x, skip_con
    
class Up(nn.Module):
    def __init__(self, in_channels, out_channels, order="middle"):
        super().__init__()

        self.upscale = GroupConvTranspose3d(in_channels, out_channels)
        self.C1 = GroupConv3d(in_channels, out_channels)
        self.dropout = Dropout(p=0.1)
        
        self.relu = nn.ReLU(inplace=True) 
        if order == "end":
            self.C2 = GroupConv3d(out_channels, out_channels, order='end')

    def forward(self, x, skip_con):
        x = self.upscale(x)

        x = torch.cat([skip_con, x], dim=2)

        x = self.C1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.C2(x)
        x = self.relu(x)

        return x


class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.C1 = GroupConv3d(in_channels, out_channels)
        self.dropout = Dropout(p=0.1)
        self.C2 = GroupConv3d(out_channels, out_channels)
        self.relu = nn.ReLU(inplace=True) 
    def forward(self, x):
        x = self.C1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.C2(x)
        x = self.relu(x)
        return x
    



class UnetGroup3d(nn.Module):
    def __init__(self):
        super(UnetGroup3d, self).__init__()

        self.down1 = Down(3, 16, order="first")
        self.down2 = Down(16, 32)
        self.down3 = Down(32, 64)

        self.bottleneck = Bottleneck(64, 128)

        self.up1 = Up(128, 64)
        self.up2 = Up(64, 32)
        self.up3 = Up(32, 16)

        self.out = nn.Conv3d(16, 4, kernel_size=(1, 1, 1))

    def forward(self, x):
        x, s1 = self.down1(x)
        s1 = s1.cpu()
        print("1")
        x, s2 = self.down2(x)
        s2 = s2.cpu()
        print("2")
        x, s3 = self.down3(x)
        print("3")
        s3 = s3.cpu()        
        x = self.bottleneck(x)
        
        s3 = s3.cuda()
        x = self.up1(x, s3)

        s2 = s2.cuda()
        x = self.up2(x, s2)

        s1 = s1.cuda()
        x = self.up3(x, s1)

        x = self.out(x)

        return x
    
if __name__ == "__main__":
    train_dataset = SegDataset("./data/train/images", "./data/train/masks")
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4)
    model = UnetGroup3d()

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scaler = GradScaler('cuda')

    num_epochs = 15

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.to(device)
    # for name, param in model.named_parameters():
    #     print(name, param.data)

    for epoch in range(num_epochs):
        model.train()
        for images, masks, in train_loader:
            optimizer.zero_grad()
            images = images.cuda()
            # masks = masks.cuda()

            with autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            print(loss.item())
        
        if (epoch + 1) % 5 == 0:
                model.cpu()
                model_save_path = f'3d_Group_UNet_Normal_{epoch+1}.pth'
                torch.save(model.state_dict(), model_save_path)
                print(f'Model saved to {model_save_path}')
                model.to(device)
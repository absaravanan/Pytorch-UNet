import torch
import torch.nn as nn
import torch.nn.functional as F


class fast_unet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super().__init__()
        self.inConv = nn.Sequential(
            nn.Conv2d(n_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.down1 = nn.Sequential(
            nn.Conv2d(64, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.upConv1 = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upConv2 = nn.Sequential(
            nn.Conv2d(256, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upConv3 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.outConv = nn.Conv2d(64, n_classes, 1)


    def concat_output(self, x1,x2):
        # print (x1.size())
        # print (x2.size())

        diffY = x2.size()[2] - x1.size()[2]
        diffX = x1.size()[3] - x1.size()[3]
        x1 = F.pad(x1, (diffX // 2, diffX - diffX//2,
                        diffY // 2, diffY - diffY//2))
        x = torch.cat([x2, x1], dim=1)
        return x

    def forward(self, x):
        x1 = self.inConv(x)
        # print (x1.size())
        x2 = self.down1(x1)
        # print (x2.size())
        x3 = self.down2(x2)
        # print (x3.size())



        x = self.up1(x3)
        # print (x.size(),1)
        # x = x + x2
        # print (x.size(),2)
        x = self.upConv1(x)
        # print (x.size(),3)
        x = x + x2
        

        x = self.up2(x)

        # x = self.concat_output(x, x2)
        # print (x.size(),4)

        x = self.upConv2(x)
               
        x = self.up3(x)
        # x = self.concat_output(x, x1)
        x = self.upConv3(x)

        x = self.outConv(x)
        return F.sigmoid(x)

if __name__ == "__main__":
    from torchsummary import summary
    net = fast_unet(n_channels=3, n_classes=1)
    summary(net, (3, 224,224))
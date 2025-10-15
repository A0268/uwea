
import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import mit_b0, mit_b1, mit_b2, mit_b3, mit_b4, mit_b5


class conv_encoder(nn.Module): #input[8, 1, 512, 512]; output[8, 8, 42, 42]
    def __init__(self):
        super(conv_encoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=3, padding=1),  # [8, 16, 171, 171]
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),  # [8, 16, 85, 85]
            nn.Conv2d(16, 8, 3, stride=2, padding=1),  # [8, 8, 43, 43]
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1)  # [8, 8, 42, 42] 
        )

    def forward(self, x):
        encode = self.encoder(x)
        return encode

class conv_decoder(nn.Module):
    def __init__(self):
        super(conv_decoder, self).__init__()

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, 3, stride=2),  # (8, 16, 85, 85)
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 5, stride=3),  # (8, 8, 257, 257)
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 2, 2, stride=2, padding=1),  # (8, num_class, 512, 512)
            #nn.Sigmoid()
        )

    def forward(self, x):
        decode = self.decoder(x)
        return decode


class MLP(nn.Module):
    """
    Linear Embedding
    """
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x
    
class ConvModule(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=0, g=1, act=True):
        super(ConvModule, self).__init__()
        self.conv   = nn.Conv2d(c1, c2, k, s, p, groups=g, bias=False)
        self.bn     = nn.BatchNorm2d(c2, eps=0.001, momentum=0.03)
        self.act    = nn.ReLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))

class SegFormerHead(nn.Module):
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
    """
    def __init__(self, num_classes=20, in_channels=[32, 64, 160, 256], embedding_dim=768, dropout_ratio=0.1):
        super(SegFormerHead, self).__init__()
        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = in_channels

        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)

        self.linear_fuse = ConvModule(
            c1=embedding_dim*4,
            c2=embedding_dim,
            k=1,
        )

        self.linear_pred    = nn.Conv2d(embedding_dim, num_classes, kernel_size=1)
        self.dropout        = nn.Dropout2d(dropout_ratio)
    
    def forward(self, inputs):
        c1, c2, c3, c4 = inputs

        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4.shape
        
        _c4 = self.linear_c4(c4).permute(0,2,1).reshape(n, -1, c4.shape[2], c4.shape[3])
        _c4 = F.interpolate(_c4, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c3 = self.linear_c3(c3).permute(0,2,1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c3 = F.interpolate(_c3, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c2 = self.linear_c2(c2).permute(0,2,1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c2 = F.interpolate(_c2, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c1 = self.linear_c1(c1).permute(0,2,1).reshape(n, -1, c1.shape[2], c1.shape[3])

        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))

        x0 = self.dropout(_c)
        x = self.linear_pred(x0)

        return x, x0  

class SegFormer(nn.Module):
    def __init__(self, backbone, nclass, pretrained = True):
        super(SegFormer, self).__init__()
        self.in_channels = {
            'b0': [32, 64, 160, 256], 'b1': [64, 128, 320, 512], 'b2': [64, 128, 320, 512],
            'b3': [64, 128, 320, 512], 'b4': [64, 128, 320, 512], 'b5': [64, 128, 320, 512],
        }[backbone]
        self.backbone   = {
            'b0': mit_b0, 'b1': mit_b1, 'b2': mit_b2,
            'b3': mit_b3, 'b4': mit_b4, 'b5': mit_b5,
        }[backbone](pretrained)
        self.embedding_dim   = {
            'b0': 256, 'b1': 256, 'b2': 768,
            'b3': 768, 'b4': 768, 'b5': 768,
        }[backbone]
        self.decode_head = SegFormerHead(nclass, self.in_channels, self.embedding_dim)

        self.conv_feat = nn.Sequential(nn.Conv2d(768, 256, kernel_size=1, bias=False), 
                                    nn.BatchNorm2d(256),
                                    nn.ReLU(True)) 
    
    
    def forward(self, inputs, need_fp=False):
        H, W = inputs.size(2), inputs.size(3)
        
        x = self.backbone.forward(inputs) 

        x0, feat = self.decode_head.forward(x)
        feat = self.conv_feat(feat) 


        if need_fp: 
            x[0] = torch.cat((x[0], nn.Dropout2d(0.5)(x[0])))
            x[1] = torch.cat((x[1], nn.Dropout2d(0.5)(x[1])))
            x[2] = torch.cat((x[2], nn.Dropout2d(0.5)(x[2])))
            x[3] = torch.cat((x[3], nn.Dropout2d(0.5)(x[3])))


            outs, _ = self.decode_head.forward(x)
            
            outs = F.interpolate(outs, size=(H, W), mode="bilinear", align_corners=True)
            out, out_fp = outs.chunk(2)
            return out, out_fp, feat   
        
        else:
            # x, _ = self.decode_head.forward(x)
            out = F.interpolate(x0, size=(H, W), mode='bilinear', align_corners=True)
            return out, feat  


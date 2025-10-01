import os
import math
import torch
import torch.nn as nn
from mamba_ssm import Mamba
from einops.layers.torch import Rearrange


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def manh_fuzzy(data, lambda_m): 
    '''
    Manhattan distance
    '''
    _, _, h, w = data.shape
    manh_weights = torch.zeros(h, w)
    center = (w - 1) // 2
    for i in range(h):
        for j in range(w):
            distance = torch.abs(torch.tensor(i - center)) + torch.abs(torch.tensor(j - center))
            # distance = torch.max(torch.abs(torch.tensor(i - center)), torch.abs(torch.tensor(j - center)))
            distance = distance.clone().detach()
            manh_weights[i, j] = torch.exp(-distance / lambda_m)
    manh_weights = manh_weights.unsqueeze(dim=0)
    manh_weights = manh_weights.unsqueeze(dim=0)
    return manh_weights.to(device)

def guss_fuzzy(data, lambda_g):
    '''
    Gaussian distance
    '''
    # lambda_g = 0.5 to 6
    _, _, h, w = data.shape
    center_index = (h // 2, w // 2)
    new_data = torch.zeros_like(data)
    for i in range(h):
        for j in range(w):
            squared_distances =(data[:, :, i, j] - data[:, :, center_index[0], center_index[1]])**2
            squared_distances = squared_distances.clone().detach()
            new_data[:, :, i, j] = squared_distances
    # data_weights = torch.exp(-h*w*new_data / (lambda_g * torch.sum(new_data)))
    data_weights = torch.exp(-new_data / (lambda_g * torch.sum(new_data)))
    return data_weights.to(device)


class fuzzy_sim(nn.Module):
    '''
    Fuzzy similarity 
    '''
    def __init__(self, lambda_g, lambda_m):
        super().__init__()
        self.lambda_g = lambda_g
        self.lambda_m = lambda_m

    def forward(self, x):
        weight_guss = guss_fuzzy(x, self.lambda_g)
        weight_manh = manh_fuzzy(x, self.lambda_m)
        out_weight = weight_guss * weight_manh
        return out_weight.to(device)


class PatchEmbeddings(nn.Module):
    def __init__(self, patch_size: int, patch_dim: int, emb_dim: int):
        super().__init__()
        self.patchify = Rearrange(
            "b c (h p1) (w p2) -> b (h w) c p1 p2",
            p1=patch_size, p2=patch_size)

        self.flatten = nn.Flatten(start_dim=2)
        self.proj = nn.Linear(in_features=patch_dim, out_features=emb_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patchify(x)
        x = self.flatten(x)
        x = self.proj(x)
        return x


class PositionalEmbeddings(nn.Module):
    def __init__(self, num_pos: int, dim: int):
        super().__init__()
        self.pos = nn.Parameter(torch.randn(num_pos, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pos


class Pooling(nn.Module): 
    def __init__(self, pool: str = "mean"):
        super().__init__()
        if pool not in ["mean", "cls"]:
            raise ValueError("pool must be one of {mean, cls}")
        self.pool_fn = self.mean_pool if pool == "mean" else self.cls_pool

    def mean_pool(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=1)

    def cls_pool(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, 0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool_fn(x)


class Classifier(nn.Module):
    def __init__(self, dim: int, num_classes: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(in_features=dim, out_features=num_classes)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class SpecAttn_small(nn.Module):
    def __init__(self, in_planes, datasetname):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        if datasetname == 'MUUFL':
            self.shared_MLP = nn.Sequential(
                nn.Conv2d(in_planes, 16, 1, bias=False), nn.BatchNorm2d(16), nn.ReLU(inplace=True),
                nn.Conv2d(16, 16, 1, bias=False), nn.BatchNorm2d(16), nn.ReLU(inplace=True),
                nn.Conv2d(16, 2, 1, bias=False), nn.BatchNorm2d(2), nn.ReLU(inplace=True)
            )
        elif datasetname == 'Augsburg':
            self.shared_MLP = nn.Sequential(
                nn.Conv2d(in_planes, 16, 1, bias=False), nn.BatchNorm2d(16), nn.ReLU(inplace=True),
                nn.Conv2d(16, 16, 1, bias=False), nn.BatchNorm2d(16), nn.ReLU(inplace=True),
                nn.Conv2d(16, 4, 1, bias=False), nn.BatchNorm2d(4), nn.ReLU(inplace=True)
            )
        elif datasetname == 'Berlin':
            self.shared_MLP = nn.Sequential(
                nn.Conv2d(in_planes, 16, 1, bias=False), nn.BatchNorm2d(16), nn.ReLU(inplace=True),
                nn.Conv2d(16, 16, 1, bias=False), nn.BatchNorm2d(16), nn.ReLU(inplace=True),
                nn.Conv2d(16, 4, 1, bias=False), nn.BatchNorm2d(4), nn.ReLU(inplace=True)
            )
        elif datasetname == 'Houston2013':
            self.shared_MLP = nn.Sequential(
                nn.Conv2d(in_planes, 16, 1, bias=False), nn.BatchNorm2d(16), nn.ReLU(inplace=True),
                nn.Conv2d(16, 16, 1, bias=False), nn.BatchNorm2d(16), nn.ReLU(inplace=True),
                nn.Conv2d(16, 8, 1, bias=False), nn.BatchNorm2d(8), nn.ReLU(inplace=True)
            )
        elif datasetname == 'Trento':
            self.shared_MLP = nn.Sequential(
                nn.Conv2d(in_planes, 16, 1, bias=False), nn.BatchNorm2d(16), nn.ReLU(inplace=True),
                nn.Conv2d(16, 16, 1, bias=False), nn.BatchNorm2d(16), nn.ReLU(inplace=True),
                nn.Conv2d(16, 1, 1, bias=False),  nn.BatchNorm2d(1), nn.ReLU(inplace=True)
            )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.shared_MLP(self.avg_pool(x))
        max_out =self.shared_MLP(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpecAttn_large(nn.Module):
    def __init__(self, in_planes, datasetname):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.shared_NET = nn.Sequential(
                nn.Conv2d(in_planes, 16, 1, bias=False), nn.BatchNorm2d(16), nn.ReLU(inplace=True),
                nn.Conv2d(16, 16, 1, bias=False), nn.BatchNorm2d(16), nn.ReLU(inplace=True),
                nn.Conv2d(16, 30, 1, bias=False), nn.BatchNorm2d(30), nn.ReLU(inplace=True)
            )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out =self.shared_NET(self.avg_pool(x))
        max_out =self.shared_NET(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatAttn(nn.Module):
    def __init__(self, in_planes):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 1, 3, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)
    

class DKFM(nn.Module):
    def __init__(self, datasetname):
        super().__init__()
        self.datasetname = datasetname
        if self.datasetname == 'MUUFL':
            in_channels_s = 2
            in_channels_l = 30
        elif self.datasetname == 'Augsburg':
            in_channels_s = 4
            in_channels_l = 30
        elif self.datasetname == 'Berlin':
            in_channels_s = 4
            in_channels_l = 30
        elif self.datasetname == 'Houston2013':
            in_channels_s = 8
            in_channels_l = 30
        elif self.datasetname == 'Trento':
            in_channels_s = 1
            in_channels_l = 30
        self.spec_s = SpecAttn_small(in_planes = in_channels_l, datasetname = self.datasetname)
        self.spec_l = SpecAttn_large(in_planes = in_channels_s, datasetname = self.datasetname)
        self.spat_s = SpatAttn(in_planes = in_channels_s)
        self.spat_l = SpatAttn(in_planes = in_channels_l)
    
    def data_slice(self, x1, x2): 
        '''
        HSI Modality: 30 Bands
        LiDAR/MSI/SAR Modality: Original Bands
        Generative Information for Each Modality: 30 Bands
        '''
        hsi_data = x1[:, :30, :, :]
        lidar_know = x1[:, 30:, :, :]
        if self.datasetname == 'MUUFL': 
            lidar_data = x2[:, :2, :, :]
            hsi_know = x2[:, 2:, :, :]
        elif self.datasetname == 'Augsburg': 
            lidar_data = x2[:, :4, :, :]
            hsi_know = x2[:, 4:, :, :]
        elif self.datasetname == 'Berlin': 
            lidar_data = x2[:, :4, :, :]
            hsi_know = x2[:, 4:, :, :]
        elif self.datasetname == 'Houston2013': 
            lidar_data = x2[:, :8, :, :]
            hsi_know = x2[:, 8:, :, :]
        elif self.datasetname == 'Trento': 
            lidar_data = x2[:, :1, :, :]
            hsi_know = x2[:, 1:, :, :]
        
        return hsi_data, hsi_know, lidar_data, lidar_know

    def forward(self, x1, x2):
        hsi_data, hsi_know, lidar_data, lidar_know = self.data_slice(x1, x2)
        source_11 = self.spec_s(hsi_know) * lidar_data
        source_12 = self.spat_s(lidar_know) * hsi_data
        source_1 = torch.cat([source_12, source_11], axis=1)
        source_21 = self.spec_l(lidar_data) * hsi_know
        source_22 = self.spat_l(hsi_data) * lidar_know
        source_2 = torch.cat([source_21, source_22], axis=1)
        return source_1, source_2


class FRConvB(nn.Module):
    def __init__(self, channels, emb_dim, lambda_g, lambda_m):
        super(FRConvB, self).__init__()

        self.lambda_g = lambda_g
        self.lambda_m = lambda_m
        self.act = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(channels, emb_dim, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(emb_dim)
        self.fuzzy_mea = fuzzy_sim(lambda_g=self.lambda_g, lambda_m=self.lambda_m)
        self.conv2 = nn.Conv2d(emb_dim, emb_dim, kernel_size=3, padding=1, groups=emb_dim, bias=False)
        self.bn3 = nn.BatchNorm2d(3*emb_dim)
        self.conv3 = nn.Conv2d(3*emb_dim, channels, kernel_size=1, bias=False)

    def forward(self, x):
        x0 = self.conv1(x)
        x0_weight = self.fuzzy_mea(x0)
        x1 = self.conv2(self.act(self.bn2(x0)))
        x2 = torch.cat([x0_weight * x0, x1, x0_weight * x0], axis=1)
        x_pre = self.conv3(self.act(self.bn3(x2))) + x
        return x_pre


class DiffViT_CLS(nn.Module):
    def __init__(self, channels, num_classes, image_size, datasetname, head_dim: int, hidden_dim: int, 
                 emb_dim: int, patch_size: int, lambda_g, lambda_m, pool: str = "mean"):
        super().__init__()
        self.datasetname = datasetname
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.channels = channels
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.num_patch = int(math.sqrt(self.num_patches))
        self.patch_dim = channels * patch_size ** 2
        self.lambda_g = lambda_g
        self.lambda_m = lambda_m
        self.act = nn.ReLU(inplace=True)

        self.d_state = 16
        self.d_conv = 4
        self.expand = 2

        # Fuzzy
        self.frconv = FRConvB(self.channels, self.emb_dim, self.lambda_g, self.lambda_m)
        # Embeddings
        self.patch_embeddings = PatchEmbeddings(patch_size=self.patch_size, patch_dim=self.patch_dim, emb_dim=self.emb_dim)
        self.pos_embeddings = PositionalEmbeddings(num_pos=self.num_patches, dim=self.emb_dim)
        # Mamba
        self.mamba = Mamba(d_model=self.emb_dim, d_state=self.d_state, d_conv=self.d_conv, expand=self.expand, bimamba_type='v2',)
        # self.mamba = Mamba(d_model=self.emb_dim, d_state=self.d_state, d_conv=self.d_conv, expand=self.expand)
        # Classifier
        self.dropout = nn.Dropout(0.8)
        self.pool = Pooling(pool=pool)
        self.classifier = Classifier(dim=emb_dim, num_classes=num_classes)

    def ViT_Mode(self, x):
        x_patch = self.patch_embeddings(x)
        x_pos = self.pos_embeddings(x_patch)
        out = self.mamba(x_pos)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_fuzzy = self.frconv(x)
        out = self.ViT_Mode(x_fuzzy)
        x_fea_2 = self.pool(self.dropout(out))
        x_cls = self.classifier(x_fea_2)
        return x_fea_2, x_cls


class MMHSIC(nn.Module):
    def __init__(self, channels_1, channels_2, num_classes, image_size, datasetname, lambda_g, lambda_m, num_views):
        super(MMHSIC, self).__init__()
        self.num_views = num_views
        self.num_classes = num_classes
        self.datasetname = datasetname

        self.lambda_g = lambda_g
        self.lambda_m = lambda_m

        head_dim = 128
        hidden_dim = 128
        emb_dim = 256
        patch_size = 1

        self.dkfms = DKFM(self.datasetname)
        self.view_net1 = DiffViT_CLS(channels_1, num_classes, image_size, datasetname, head_dim, 
                                     hidden_dim, emb_dim, patch_size, lambda_g, lambda_m)
        self.view_net2 = DiffViT_CLS(channels_2, num_classes, image_size, datasetname, head_dim, 
                                     hidden_dim, emb_dim, patch_size, lambda_g, lambda_m)
        

    def forward(self, x1, x2):
        view1_x, view2_x = self.dkfms(x1, x2)
        view1_fea_2, view1_cls = self.view_net1(view1_x)
        view2_fea_2, view2_cls = self.view_net2(view2_x)
        view_fea_2_all = torch.cat([view1_fea_2, view2_fea_2], dim=0)
        view_cls_all = torch.cat([view1_cls, view2_cls], dim=0)
        return view_fea_2_all, view_cls_all


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # input1: [batch_size, cat[hsi_data(30), lidar_diff(2)/sar_diff(4)/msi_diff(8)], patch_size, patch_size]
    # input2: [batch_size, cat[lidar_data(2)/sar_data(4)/msi_data(8), hsi_diff(30)], patch_size, patch_size]
    input1 = torch.randn(size=(64, 32, 11, 11)).to(device)
    input2 = torch.randn(size=(64, 32, 11, 11)).to(device)
    print("input 1 shape:", input1.shape)
    print("input 2 shape:", input2.shape)
    model = MMHSIC(channels_1=32, channels_2=32, num_classes=11, image_size=11, datasetname='MUUFL', 
                   lambda_g=0.5, lambda_m=11, num_views=2).to(device) # lambda_m = 1 * image_size
    out_fea, out_cls = model(input1, input2)
    print("Feature shape:", out_fea.shape)
    print("Class shape:", out_cls.shape)

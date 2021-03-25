import torch.nn as nn
import torch

class Attr_Decoder_conv(nn.Module):
    def __init__(self, za_size, attribute_dim):
        super(Attr_Decoder_conv, self).__init__()
        self.layers = nn.ModuleList()
        hidden_dim = 1024
        self.first_layer = nn.Linear(za_size, hidden_dim)
        for layer_index in range(4):
            conv_block = nn.Sequential(
                nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim//2, kernel_size=3, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(hidden_dim//2)
            )
            hidden_dim = hidden_dim // 2   # 512-256-128-64
            self.layers.append(conv_block)
        last_block = nn.Sequential(
            nn.Conv1d(in_channels=hidden_dim, out_channels=attribute_dim, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(attribute_dim)
        )
        self.layers.append(last_block)

    def forward(self, attr_en):
        attr = self.first_layer(attr_en)
        attr = attr.unsqueeze(2)
        for i in range(5):
            attr = self.layers[i](attr)
        attr = attr.squeeze(2)
        return attr


class cam_Classifier(nn.Module):
    def __init__(self, embed_dim, cam_class):
        super(cam_Classifier, self).__init__()
        hidden_size = 2048
        self.first_layer = nn.Sequential(
            nn.Conv1d(in_channels=embed_dim, out_channels=hidden_size, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True)
        )
        self.layers = nn.ModuleList()
        for layer_index in range(5):
            conv_block = nn.Sequential(
                nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size // 2, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm1d(hidden_size // 2),
                nn.ReLU(inplace=True)

            )
            hidden_size = hidden_size // 2  # 512-32
            self.layers.append(conv_block)
        self.Liner = nn.Linear(hidden_size, cam_class)

    def forward(self, latent):
        latent = latent.unsqueeze(2)
        hidden = self.first_layer(latent)
        for i in range(5):
            hidden = self.layers[i](hidden)
        hidden = hidden.squeeze(2)
        domain_class = self.Liner(hidden)
        # domain_clss = torch.sigmoid(domain_clss)
        return domain_class, hidden  # [batch,15]

class cam_Classifier_1024(nn.Module):
    def __init__(self, embed_dim, cam_class):
        super(cam_Classifier_1024, self).__init__()
        hidden_size = 1024
        self.first_layer = nn.Sequential(
            nn.Conv1d(in_channels=embed_dim, out_channels=hidden_size, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True)
        )
        self.layers = nn.ModuleList()
        for layer_index in range(6):
            conv_block = nn.Sequential(
                nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size // 2, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm1d(hidden_size // 2),
                nn.ReLU(inplace=True)
            )
            hidden_size = hidden_size // 2  # 512-32
            self.layers.append(conv_block)
        self.Liner = nn.Linear(hidden_size, cam_class)

    def forward(self, latent):
        latent = latent.unsqueeze(2)
        hidden = self.first_layer(latent)
        for i in range(6):
            hidden = self.layers[i](hidden)
        hidden = hidden.squeeze(2)
        domain_class = self.Liner(hidden)
        # domain_clss = torch.sigmoid(domain_clss)
        return domain_class, hidden  # [batch,15]

class cam_Classifier_1024_nobias(nn.Module):
    def __init__(self, embed_dim, cam_class):
        super(cam_Classifier_1024_nobias, self).__init__()
        hidden_size = 1024
        self.first_layer = nn.Sequential(
            nn.Conv1d(in_channels=embed_dim, out_channels=hidden_size, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True)
        )
        self.layers = nn.ModuleList()
        for layer_index in range(6):
            conv_block = nn.Sequential(
                nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size // 2, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm1d(hidden_size // 2),
                nn.ReLU(inplace=True)
            )
            hidden_size = hidden_size // 2  # 512-32
            self.layers.append(conv_block)
        self.Liner = nn.Linear(hidden_size, cam_class, bias=False)

    def forward(self, latent):
        latent = latent.unsqueeze(2)
        hidden = self.first_layer(latent)
        for i in range(6):
            hidden = self.layers[i](hidden)
        hidden = hidden.squeeze(2)
        domain_class = self.Liner(hidden)
        # domain_clss = torch.sigmoid(domain_clss)
        return domain_class, hidden  # [batch,15]

class cam_Classifier_fc(nn.Module):
    def __init__(self, embed_dim, cam_class):
        super(cam_Classifier_fc, self).__init__()
        hidden_size = 1024
        self.first_layer = nn.Sequential(
            nn.Linear(embed_dim, hidden_size, bias=False),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True)
        )
        self.layers = nn.ModuleList()
        for layer_index in range(6):
            conv_block = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2, bias=False),
                nn.BatchNorm1d(hidden_size // 2),
                nn.ReLU(inplace=True)
            )
            hidden_size = hidden_size // 2  # 512-32
            self.layers.append(conv_block)
        self.Liner = nn.Linear(hidden_size, cam_class)

    def forward(self, latent):
        hidden = self.first_layer(latent)
        for i in range(6):
            hidden = self.layers[i](hidden)
        domain_class = self.Liner(hidden)
        # domain_clss = torch.sigmoid(domain_clss)
        return domain_class, hidden  # [batch,15]

class cam_Classifier_fc_nobias_in_last_layer(nn.Module):
    def __init__(self, embed_dim, cam_class):
        super(cam_Classifier_fc_nobias_in_last_layer, self).__init__()
        hidden_size = 1024
        self.first_layer = nn.Sequential(
            nn.Linear(embed_dim, hidden_size, bias=False),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True)
        )
        self.layers = nn.ModuleList()
        for layer_index in range(6):
            conv_block = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2, bias=False),
                nn.BatchNorm1d(hidden_size // 2),
                nn.ReLU(inplace=True)
            )
            hidden_size = hidden_size // 2  # 512-32
            self.layers.append(conv_block)
        self.Liner = nn.Linear(hidden_size, cam_class, bias=False)

    def forward(self, latent):
        hidden = self.first_layer(latent)
        for i in range(6):
            hidden = self.layers[i](hidden)
        domain_class = self.Liner(hidden)
        # domain_clss = torch.sigmoid(domain_clss)
        return domain_class, hidden  # [batch,15]



class CamClassifier(nn.Module):
    def __init__(self, in_dim, target_cam):
        super(CamClassifier, self).__init__()
        self.in_dim = in_dim
        # self.source_cam = source_cam
        self.target_cam = target_cam

        self.layer1 = self._make_layer(self.in_dim, 1024)
        self.layer2 = self._make_layer(1024, 512)
        self.layer3 = self._make_layer(512, 256)
        self.layer4 = self._make_layer(256, 128)
        self.layer5 = self._make_layer(128, 64)
        self.layer6 = self._make_layer(64, 32)
        self.fc = nn.Linear(32,  target_cam)

    def _make_layer(self, in_nc, out_nc):
        block = [nn.Conv2d(in_nc, out_nc, kernel_size=3, stride=1, padding=1),
                 nn.BatchNorm2d(out_nc),
                 nn.LeakyReLU(0.2, True)]
        return nn.Sequential(*block)

    def forward(self, x):
        output = self.layer6(self.layer5(self.layer4(self.layer3(self.layer2(self.layer1(x))))))
        output = self.fc(output.squeeze())
        return output

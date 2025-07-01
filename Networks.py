import torch
from torch import nn
import torch.nn.functional as F
from torchvision.ops import MLP

class Generator(nn.Module):
    def __init__(self, input_dim):
        super(Generator, self).__init__()

        # content encoder
        self.enc_content = ContentEncoder(input_dim)

        # style encoder
        self.enc_style = StyleEncoder(input_dim)

        # decoder
        self.dec = Decoder(self.enc_content.output_dim, input_dim)

        # get AdaIN parameters
        self.mlp = MLP(self.enc_style.output_dim, self.get_num_adain_params(self.dec))

    # encode an image into its content and style codes
    def encode(self, img):
        content = self.enc_content(img)
        style = self.enc_style(img)
        return content, style

    # decode content and style codes to an image
    def decode(self, content, style):
        adain_params = self.mlp(style)
        self.assign_adain_params(adain_params, self.dec)
        img = self.dec(content)
        return img

    def assign_adain_params(self, adain_params, model):
        # assign the adain_params to the AdaIN layers in model
        for m in model.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                mean = adain_params[:, :m.num_features]
                std = adain_params[:, m.num_features:2 * m.num_features]
                m.bias = mean.contiguous().view(-1)
                m.weight = std.contiguous().view(-1)
                if adain_params.size(1) > 2 * m.num_features:
                    adain_params = adain_params[:, 2 * m.num_features:]

    def get_num_adain_params(self, model):
        # return the number of AdaIN parameters needed by the model
        num_adain_params = 0
        for m in model.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                num_adain_params += 2 * m.num_features
        return num_adain_params

    def forward(self, img):
        content, style = self.encode(img)
        img_recon = self.decode(content, style)
        return img_recon


class ContentEncoder(nn.Module):
    def __init__(self, input_dim):
        super(ContentEncoder, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(input_dim, 32, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

        self.output_dim = 64

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def forward_3(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        return x


class StyleEncoder(nn.Module):
    def __init__(self, input_dim):
        super(StyleEncoder, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(input_dim, 32, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.output_dim = 128

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


class Decoder(nn.Module):
    def __init__(self, feature_dim, output_dim):
        super(Decoder, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=feature_dim, out_channels=feature_dim // 2, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            FFBlock(feature_dim // 2),
            FFBlock(feature_dim // 2)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=feature_dim // 2, out_channels=output_dim, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x


class MLP_classifier(nn.Module):
    def __init__(self, input_dim):
        super(MLP_classifier, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.Linear(256, 32),
            nn.Linear(32, 2),
        )

        self.flatten = nn.Flatten()

    def forward(self, c):
        c = self.flatten(c)
        out = self.fc(c)
        return out


class CD_encoder(nn.Module):
    def __init__(self, input_dim=32, output_dim=16):
        super(CD_encoder, self).__init__()

        self.layer = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        out = self.layer(x)
        return out


class DIFM(nn.Module):
    def __init__(self, input_dim=16):
        super(DIFM, self).__init__()
        self._SAM_a = SpatialAttentionModule()
        self._DCA_a = DCAttention(input_dim)
        self._SAM_b = SpatialAttentionModule()
        self._DCA_b = DCAttention(input_dim)

    def forward(self, c_a, c_ba, c_b, c_ab):

        # Attention map generation
        cam_map_a = self._DCA_a(torch.abs(c_a - c_ba))
        cam_map_b = self._DCA_b(torch.abs(c_b - c_ab))

        # Domain_a
        c_a_fusion = torch.cat((c_a, c_ba), dim=1)
        c_a_fusion = c_a_fusion * cam_map_b + c_a_fusion

        # Domain_b
        c_b_fusion = torch.cat((c_b, c_ab), dim=1)
        c_b_fusion = c_b_fusion * cam_map_a + c_b_fusion

        DI_fusion = c_a_fusion + c_b_fusion

        return DI_fusion


class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return out


class DCAttention(nn.Module):
    def __init__(self, in_channels, kernel_size=3):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.conv1d = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
        self.channel_expand = nn.Conv1d(in_channels, in_channels * 2, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        B, C, H, W = x.size()

        y = self.gap(x)
        y = y.view(B, C, -1).permute(0, 2, 1)

        y = self.conv1d(y)

        y = y.permute(0, 2, 1)
        y = self.channel_expand(y)
        y = y.unsqueeze(-1)

        y = self.sigmoid(y)

        return y


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP, self).__init__()

        hidden_dim = input_dim * 4

        self.fc1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.5)
        )

        self.fc2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

        self.fc3 = nn.Sequential(
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


class FFBlock(nn.Module):
    def __init__(self, dim):
        super(FFBlock, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1),
            AdaptiveInstanceNorm2d(dim),
            nn.GELU(),
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1),
            AdaptiveInstanceNorm2d(dim)
        )

    def forward(self, x):
        residual = x
        x = self.layer1(x)
        x = self.layer2(x)
        x += residual
        return x


class AdaptiveInstanceNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(AdaptiveInstanceNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        # weight and bias are dynamically assigned
        self.weight = None
        self.bias = None
        # just dummy buffers, not used
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x):
        assert self.weight is not None and self.bias is not None, "Please assign weight and bias before calling AdaIN!"
        b, c = x.size(0), x.size(1)
        running_mean = self.running_mean.repeat(b)
        running_var = self.running_var.repeat(b)

        # Apply instance norm
        x_reshaped = x.contiguous().view(1, b * c, *x.size()[2:])

        out = F.batch_norm(
            x_reshaped, running_mean, running_var, self.weight, self.bias,
            True, self.momentum, self.eps)

        return out.view(b, c, *x.size()[2:])

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.num_features) + ')'


import torch
import torch.nn as nn

def Conv(ch_in, ch_out, kernel_size, stride, padding):
    return nn.Sequential(
        nn.Conv2d(ch_in, ch_out, kernel_size, stride, padding, bias=False),
        nn.InstanceNorm2d(ch_out, affine=True),
        nn.LeakyReLU(0.2)
    )

def DeConv(ch_in, ch_out, kernel_size, stride, padding):
    return nn.Sequential(
        nn.ConvTranspose2d(ch_in, ch_out, kernel_size, stride, padding, bias=False),
        nn.BatchNorm2d(ch_out),
        nn.ReLU()
    )


class Discriminator(nn.Module):
    def __init__(self, p_min, p_max, ch_img, ch_in):
        super(Discriminator, self).__init__()
        self.layers = []

        #first section (output: n x ch_in x 16)
        self.layers.extend([
                nn.Conv2d(ch_img, ch_in, kernel_size=4, stride=2, padding=1),
                nn.LeakyReLU(0.2),
            ])

        #middle section
        for i in range(p_min, p_max-1):
            self.layers.append(
                Conv(ch_in * 2 ** (i-p_min), ch_in * 2 ** (i-p_min+1), kernel_size=4, stride=2, padding=1)
            )

        # last section
        ch_out = ch_in * 2 ** (p_max-p_min-1)
        self.layers.extend([
            nn.Conv2d(ch_out, 1, kernel_size=4, stride=2, padding=0),
        ])

        self.Sequential = nn.Sequential(*self.layers)
            
    
    def forward(self, x_b):
        return self.Sequential(x_b)


class Generator(nn.Module):
    def __init__(self, p_min, p_max, z_size, ch_img, ch_in):
        super(Generator, self).__init__()
        self.layers = []

        #first section
        self.layers.append(
            DeConv(z_size, ch_in, 4, 1, 0)
        )

        #middle section
        for i in range(p_min, p_max-1):
            self.layers.append(
                DeConv(ch_in // 2 ** (i-p_min), ch_in // 2 ** (i-p_min+1), kernel_size=4, stride=2, padding=1)
            )

        #last section
        ch_out = ch_in // 2 ** (p_max-p_min-1)
        self.layers.extend([
            nn.ConvTranspose2d(ch_out, ch_img, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        ])

        self.Sequential = nn.Sequential(*self.layers)

    def forward(self, x_b):
        return self.Sequential(x_b)

def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)


def main():
    z_size = 128
    p_min = 2
    p_max = 5

    g_ch_in = 512
    d_ch_in = g_ch_in // 2 ** (p_max-p_min-1)

    G = Generator(p_min, p_max, z_size, 1, g_ch_in)

    print(f"Discriminator output: {Discriminator(p_min, p_max, 1, d_ch_in)(torch.randn(10, 1, 32, 32)).shape}")
    print(f"Generator output: {G(torch.randn(10, z_size, 1, 1)).shape}")

    z = torch.randn(100, 128, 1, 1)
    D_fake = G(z).reshape(-1)
    print(min(D_fake), max(D_fake))
    nn.BCELoss()(D_fake, torch.zeros_like(D_fake))
    print("success")

if __name__ == "__main__":
    main()
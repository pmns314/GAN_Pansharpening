import torch.nn as nn
import torch


class RaganLoss(nn.Module):
    def __init__(self):
        super(RaganLoss, self).__init__()
        self.mse = nn.MSELoss(reduction="mean")
        self.ones = None
        self.use_real_data = True
        self.fake_label = 1  # Label for Fake Data
        self.real_label = 0  # Label for Real Data

    def ragan_loss(self, x1, x2):

        # L_RaGAN(x1, x2) = loss( D(x1) - mean( D(x2) )
        loss_fake = self.mse(x1 - torch.mean(x2), self.ones * self.fake_label)
        loss_real = self.mse(x2 - torch.mean(x1), self.ones * self.real_label)

        return (loss_real + loss_fake) / 2

    def forward(self, fake, real, is_generator=False):
        if self.ones is None or self.ones.shape != fake.shape:
            self.ones = torch.ones_like(fake)

        if is_generator:
            return self.ragan_loss(fake, real)
        else:
            return self.ragan_loss(real, fake)


class LsganLoss(nn.Module):
    def __init__(self):
        super(LsganLoss, self).__init__()
        self.mse = nn.MSELoss(reduction="mean")
        self.ones = None
        self.use_real_data = False
        self.fake_label = 1  # Label for Fake Data
        self.real_label = 0  # Label for Real Data

    def generator_loss(self, fake):
        # Generator
        # If the discriminator classifies correctly, it's an error for the generator
        return self.mse(fake, self.ones * self.fake_label)

    def discriminator_loss(self, fake, real):
        # Discriminator
        # min(1/2 * E[ (D(x) - b)^2 ] + 1/2 * E[ (D(G(z)) - a)^2 ])
        fake_prob = self.mse(fake, self.ones * self.real_label)  # E[ (D(G(z)) - a)^2 ])
        true_prob = self.mse(real, self.ones * self.fake_label)  # E[ (D(x) - b)^2 ]
        tot_prob = true_prob + fake_prob
        return tot_prob * 0.5

    def forward(self, fake, real, is_generator=False):
        if self.ones is None or self.ones.shape != fake.shape:
            self.ones = torch.ones_like(fake)

        if is_generator:
            return self.generator_loss(fake)
        else:
            return self.discriminator_loss(fake, real)


class MinimaxLoss(nn.Module):
    def __init__(self):
        super(MinimaxLoss, self).__init__()
        self.EPS = 1e-12
        self.use_real_data = False

    def generator_loss(self, fake):
        # Generator
        # min( E[ log(1 - D(G(z))) ]) ---> max ( E[ log(D(G(z))) ] )
        fake_prob = torch.log(fake + self.EPS)  # log(D(G(z)))
        tot_prob = fake_prob
        return torch.mean(-tot_prob)

    def discriminator_loss(self, fake, real):
        # Discriminator
        # max(E[ log(D(x) ] + E[ log(1 - D(G(z))) ])
        fake_prob = torch.log(1 - fake + self.EPS)  # log(1 - D(G(z)))
        true_prob = torch.log(real + self.EPS)  # log(D(x)
        tot_prob = true_prob + fake_prob
        return torch.mean(-tot_prob)

    def forward(self, fake, real, is_generator=False):
        if is_generator:
            return self.generator_loss(fake)
        else:
            return self.discriminator_loss(fake, real)

import torch
import torch.nn as nn


class LsganLoss(nn.Module):
    def __init__(self, use_soft_labels=False):
        super(LsganLoss, self).__init__()
        self.mse = nn.MSELoss(reduction="mean")
        self.ones = None
        self.use_real_data = False
        self.soft_labes = use_soft_labels
        self.fake_label = 1  # Label for Fake Data
        self.real_label = 0  # Label for Real Data

    def generate_soft_labels(self):
        self.fake_label = torch.distributions.uniform.Uniform(.7, 1.2).sample([1, 1])
        self.real_label = torch.distributions.uniform.Uniform(0, .3).sample([1, 1])

    def generator_loss(self, fake):
        # Generator
        # If the discriminator classifies correctly, it's an error for the generator
        return self.mse(fake, self.ones * self.fake_label) * 0.5

    def discriminator_loss(self, fake, real):
        # Discriminator
        # min(1/2 * E[ (D(x) - b)^2 ] + 1/2 * E[ (D(G(z)) - a)^2 ])
        fake_prob = self.mse(fake, self.ones * self.real_label)  # E[ (D(G(z)) - a)^2 ])
        true_prob = self.mse(real, self.ones * self.fake_label)  # E[ (D(x) - b)^2 ]
        tot_prob = true_prob + fake_prob
        return tot_prob / 2

    def forward(self, fake, real, is_generator=False):
        if self.ones is None or self.ones.shape != fake.shape:
            self.ones = torch.ones_like(fake)

        if self.soft_labes:
            self.generate_soft_labels()

        if is_generator:
            return self.generator_loss(fake)
        else:
            return self.discriminator_loss(fake, real)

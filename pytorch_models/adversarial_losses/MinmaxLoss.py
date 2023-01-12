from abc import ABC

import torch
import torch.nn as nn


class MinmaxLoss(nn.Module):
    def __init__(self):
        super(MinmaxLoss, self).__init__()
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

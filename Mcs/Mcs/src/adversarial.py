import torch
import numpy as np
import torch.nn as nn


class adv(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.device = config['device']
        self.generator = self.Generator(config)
        self.discriminator = self.Discriminator(config)


    def forward(self, batch):
        city = batch[0]
        policy = batch[1]
        similarity = finder(policy)
        similar_policy = max(similarity)
        generated_policy = self.generator(policy)
        valid = self.discriminator(generated_policy)
        adversarial_loss = torch.nn.BCELoss(generated_policy, valid)

        return adversarial_loss



    class Generator(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.config = config
            self.latent_dim = config['latent_dim']

            def block(in_feat, out_feat, normalize=True):
                self.shape = in_feat.shape
                layers = [nn.Linear(in_feat, out_feat)]
                if normalize:
                    layers.append(nn.BatchNorm1d(out_feat, 0.8))
                layers.append(nn.LeakyReLU(0.2, inplace=True))  # SeLU, GeLU, ReLU, Tanh,
                return layers

            self.model = nn.Sequential(
                *block(self.latent_dim, 128, normalize=False),
                *block(128, 256),
                *block(256, 512),
                *block(512, 1024),
                nn.Linear(1024, int(np.prod(self.shape))),
                nn.Tanh()
            )

        def forward(self, z):
            generated_policy = self.model(z)
            return generated_policy

    class Discriminator(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.config = config
            # sequential不能做端到端（这个输出是另外一个的输入）架构，只能用函数做端到端
            self.model = nn.Sequential(
                nn.Linear(1024, 512),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(512, 256),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(256, 1),
                nn.Sigmoid(),
            )

        # img_size（0）维度是batch_size
        def forward(self, img):
            img_flat = img.view(img.size(0), -1)
            validity = self.model(img_flat)

            return validity


def finder(policies):
    size = policies.shape
    size[1] = 1
    policy_mat = torch.split(policies, size, dim=1)
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    similarity_mat = []
    for policy in policy_mat:
        similarity = cos(policy, policy_mat)
        similarity_mat.append(similarity)
    return similarity_mat







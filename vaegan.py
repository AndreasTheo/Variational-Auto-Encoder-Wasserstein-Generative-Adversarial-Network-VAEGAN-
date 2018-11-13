from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib

latent_dim = 8
batch_size = 256
load_weights = False

class Encoder_Network(nn.Module):
    def __init__(self, input_dim, latent_dim):
        h = 64
        h2 = 32
        super(Encoder_Network, self).__init__()
        self.l1 = nn.Linear(input_dim, h)
        self.l2 = nn.Linear(h, h2)
        self.l3 = nn.Linear(h2, latent_dim)
        self.h_log_sigma = nn.Linear(h2, latent_dim)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        mu = self.l3(x)
        log_sigma = self.h_log_sigma(x)
        return mu, log_sigma

    def reparameterization_trick(self, latent_mu,latent_log_sigma):
        eps = torch.from_numpy(np.random.normal(0, 1, size=latent_log_sigma.size())).float()
        z = latent_mu + torch.exp(latent_log_sigma / 2) * eps
        return z

class Decoder_Network(nn.Module):
    def __init__(self, latent_dim, output_dim):
        h = 32
        h2 = 64
        super(Decoder_Network, self).__init__()
        self.l1 = nn.Linear(latent_dim, h)
        self.l2 = nn.Linear(h, h2)
        self.l3 = nn.Linear(h2, output_dim)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        output = F.sigmoid(self.l3(x))
        return output


class Discriminator_Network(nn.Module):
    def __init__(self, input_dim, output_dim):
        h = 64
        h2 = 64
        super(Discriminator_Network, self).__init__()
        self.l1 = nn.Linear(input_dim, h)
        self.l2 = nn.Linear(h, h2)
        self.l3 = nn.Linear(h2, output_dim)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        output = F.sigmoid(self.l3(x))
        return output

mnist = input_data.read_data_sets('../../MNIST_data', one_hot=True)
encoder = Encoder_Network(mnist.train.images.shape[1], latent_dim)
recon_encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-3)
kl_encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=3e-4)
maximizing_discrim_output_encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=3e-4)

decoder = Decoder_Network(latent_dim, mnist.train.images.shape[1])
recon_decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=1e-3)
kl_decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=3e-4)
maximizing_discrim_output_decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=3e-4)

discriminator = Decoder_Network(mnist.train.images.shape[1], 1)
discriminator_optimizer = torch.optim.Adam(decoder.parameters(), lr=3e-4)

real_targets = Variable(torch.from_numpy(np.ones((batch_size,1)))).float()
fake_targets = Variable(torch.from_numpy(np.zeros((batch_size,1)))).float()

bce_criterion = nn.BCELoss()
bce_criterion.size_average = False

epochs = 5000
if load_weights:
    encoder.load_state_dict(torch.load('encoder.pt'))
    decoder.load_state_dict(torch.load('decoder.pt'))
    epochs = 1 #last run for visualizations

discrim_loss = 0
reconstruction_loss = 0
kl_loss = 0
maximizing_discrim_output_loss = 0

for i in range(epochs):
    if i % 250 == 0:
        print('discrim_loss: ', discrim_loss,
              'reconstruction_loss:', reconstruction_loss,
              'kl_loss: ', kl_loss ,
              'maximizing_discrim_output_loss: ', maximizing_discrim_output_loss)

        print('training iter: ', i)
    if i == epochs-1:
        print('last')
        batch_size = 60000
    X, Y = mnist.train.next_batch(batch_size)
    Y = np.argmax(Y, axis=1)

    X = Variable(torch.from_numpy(X))
    latent_mu, latent_log_sigma = encoder(X)
    latent_sigma = torch.exp(latent_log_sigma)
    z = encoder.reparameterization_trick(latent_mu,latent_log_sigma)
    X_hat = decoder(z)

    if i < epochs - 1:
        for k in range(3):
            discriminated_fakes = discriminator(X_hat.detach())
            discriminated_reals = discriminator(X)
            discrim_loss = bce_criterion(discriminated_reals,real_targets)/batch_size + bce_criterion(discriminated_fakes,fake_targets)/batch_size

            discriminator_optimizer.zero_grad()
            discrim_loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 0.01)
            discriminator_optimizer.step()

        reconstruction_loss = (bce_criterion(X_hat,X)/batch_size)
        kl_loss = ((0.5 * torch.sum(latent_mu ** 2 + latent_sigma - latent_log_sigma - 1, 1)).mean())
        maximizing_discrim_output_loss = -discriminator(X_hat).mean()
        vae_loss = reconstruction_loss + kl_loss

        def optimize_vae_with_loss(encoder_optimizer,decoder_optimizer,loss):
            encoder_optimizer.zero_grad()
            loss.backward(retain_graph=True)
            encoder_optimizer.step()

            decoder_optimizer.zero_grad()
            loss.backward(retain_graph=True)
            decoder_optimizer.step()


        optimize_vae_with_loss(recon_encoder_optimizer,recon_decoder_optimizer,reconstruction_loss)
        optimize_vae_with_loss(kl_encoder_optimizer,kl_decoder_optimizer,kl_loss)
        optimize_vae_with_loss(maximizing_discrim_output_encoder_optimizer,
                               maximizing_discrim_output_decoder_optimizer,
                               maximizing_discrim_output_loss)


if load_weights == False:
    print('saving weights... ')
    torch.save(encoder.state_dict(), 'encoder.pt')
    torch.save(decoder.state_dict(), 'decoder.pt')

#visualize blending
fig=plt.figure(figsize=(8, 8))
columns = 4
rows = 5

blend_zero = 1
blend_one = 0

for i in range(len(z)):
    z[i] = z[0]*blend_zero + z[20]*blend_one
    blend_zero -= 0.05
    blend_one += 0.05
X_hat = decoder(z)
for i in range(1, columns*rows + 1):
    fig.add_subplot(rows, columns, i)
    plt.imshow(X_hat.data[i].numpy().reshape(28, 28))
plt.show()

# #visualize latent space
# for i in range(6):
#     latent_mu_single_code = latent_mu[:, i]
#     latent_log_sigma_single_code = latent_log_sigma[:, i]
#     plt.subplot(2, 3, i+1)
#     cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["red","violet","blue","orange","grey","pink","yellow","black","green"])
#     plt.scatter(latent_mu_single_code.data.numpy(), latent_log_sigma_single_code.data.numpy(), c=Y,cmap=cmap)
# plt.show()











from __future__ import print_function
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import torch
import torch.utils.data
from matplotlib import pyplot as plt
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image

class Autoencoder(nn.Module):
    def __init__(
            self,
            autoencoder_type="ae",
            n_dims_code=2,
            n_dims_data=784,
            hidden_layer_size=512,
            device="mps"):
        """
        Initializes the Autoencoder class with the provided parameters.

        Args:
            autoencoder_type (str): Type of the autoencoder. Either "ae" for standard autoencoder or "vae" for variational autoencoder. Defaults to "ae".
            n_dims_code (int): Dimension of the latent space (code). Defaults to 2.
            n_dims_data (int): Dimension of the input data. Defaults to 784.
            hidden_layer_size (int): Number of units in the hidden layer. Defaults to 512.
            device (str): Device to use for computations. Either "cuda" or "cpu". Defaults to "cuda".
        """
        super(Autoencoder, self).__init__()
        self.autoencoder_type = autoencoder_type
        self.n_dims_data = n_dims_data
        self.n_dims_code = n_dims_code
        self.device = device

        # TODO: Encoder layers
        self.encoder_shared = nn . Sequential (
            nn.Linear(784,512),
            nn.ReLU(),
        )
        self.encoder_mu = nn.Linear(512,2)
        self.encoder_logvar = nn.Linear(512,2)

        # TODO: Decoder layers
        self.decoder = nn.Sequential(
            nn.Linear(2,512),
            nn.ReLU(),
            nn.Linear(512,784),
            nn.Sigmoid()
        )

    def encode(self, x):
        """
        Encodes the input data.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            for AE:
                torch.Tensor: Encoded data.
            for VAE:
                torch.Tensor: mean of the latent code.
                torch.Tensor: log variance of the latent code.
        """
        # TODO
        share_encoded = self.encoder_shared(x)
        if self.autoencoder_type == "ae":
            return self.encoder_mu(share_encoded)
        elif self.autoencoder_type == "vae" :
            mu = self.encoder_mu(share_encoded)
            logvar = self.encoder_logvar(share_encoded)
            return mu,logvar

    def sample(self, mu, logvar):
        """
        Reparameterizes the encoded data for the variational autoencoder.

        Args:
            mu (torch.Tensor): Mean of the latent code.
            logvar (torch.Tensor): Log variance of the latent code.

        Returns:
            torch.Tensor: Sampled latent code.
        """
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        """
        Decodes the latent code back to the original data space.

        Args:
            z (torch.Tensor): Latent code.

        Returns:
            torch.Tensor: Reconstructed data.
        """
        # TODO
        return self.decoder(z)

    def forward(self, x):
        """
        Perform model forward.

        Args:
            x (torch.Tensor): Input data.

        Returns:
            torch.Tensor: Reconstructed data.
            torch.Tensor: Mean of the latent code.
            torch.Tensor: Log variance of the latent code.
        """
        if self.autoencoder_type == "vae":
            mu, logvar = self.encode(x.view(-1, self.n_dims_data))
            z = self.sample(mu, logvar)
        else:
            z = self.encode(x.view(-1, self.n_dims_data))
            mu = z
            logvar = None
        return self.decode(z), mu, logvar

    def loss_function(self, recon_x, x, mu, logvar):
        """
        Computes the loss function for the autoencoder.

        Args:
            recon_x (torch.Tensor): Reconstructed data.
            x (torch.Tensor): Original data.
            mu (torch.Tensor): Mean of the latent code.
            logvar (torch.Tensor): Log variance of the latent code.

        Returns:
            torch.Tensor: Total loss.
            torch.Tensor: Reconstruction loss (Binary Cross Entropy).
            torch.Tensor: KL Divergence loss (only for "vae").
        """
        BCE = F.binary_cross_entropy(recon_x, x.view(-1, self.n_dims_data), reduction='sum')
        if self.autoencoder_type == "vae":
            # TODO: compute KL divergence loss for VAE
            KLD = 1/2*(torch.sum(mu.pow(2)+logvar.exp()-logvar-1))
        else:  # If AE, there's no KL divergence
            KLD = torch.zeros(1, dtype=torch.float, device=self.device)
        return BCE + KLD, BCE, KLD

def train(epoch, model, optimizer, train_loader):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss, _, _ = model.loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader),
                       loss.item() / len(data)))

    train_loss /= len(train_loader.dataset)
    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss))
    return train_loss

def test(epoch, model, test_loader, DIR):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            loss, _, _ = model.loss_function(recon_batch, data, mu, logvar)
            test_loss += loss.item()
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n],
                                        recon_batch.view(args.batch_size, 1, 28, 28)[:n]])
                save_image(comparison.cpu(), DIR / ('reconstruction_' + str(epoch) + '.png'), nrow=n)

    test_loss /= len(test_loader.dataset)
    return test_loss


def _set_latent_vectors(z_range=2, nrows=20, ncols=20):
    z = np.rollaxis(
        np.mgrid[z_range:-z_range:nrows * 1j, z_range:-z_range:ncols * 1j], 0, 3)
    z = z.reshape([-1, 2])
    return z

def save_scattered_image(z, id, DIR, z_range=2, name='manifold.png'):
    N = 10
    plt.figure(figsize=(8, 6))
    plt.scatter(z[:, 0], z[:, 1], c=np.argmax(id, 1), marker='o', edgecolor='none', cmap=plt.get_cmap('jet', N))
    plt.colorbar(ticks=range(N))
    axes = plt.gca()
    axes.set_xlim([-z_range - 2, z_range + 2])
    axes.set_ylim([-z_range - 2, z_range + 2])
    plt.grid(True)
    plt.savefig(DIR / name)


def plot_manifold_learning_result(model, DIR, z_range=2):
    """
    This function plots the manifold learning result. It generates and saves scattered images
    of the latent vectors and their corresponding decoded images.

    Args:
        args: The command-line arguments.
        model: The trained VAE or AE model.
    """
    assert model.n_dims_code == 2
    # Set the latent vectors
    z = _set_latent_vectors()
    z_ = torch.from_numpy(z).float().to(device)

    # Decode the latent vectors
    y = model.decode(z_)

    # Reshape the decoded images for saving
    y_img = y.reshape(-1, int(np.sqrt(model.n_dims_data)), int(np.sqrt(model.n_dims_data)))
    # res = y_img.mul(255).add(0.5).clamp(0, 255).to('cpu', torch.int8).numpy()
    y_img = y_img.unsqueeze(1)

    # Save the decoded images
    save_image(y_img, DIR / 'manifold_digits.jpg', nrow=20)

    # Load the test dataset
    test_dataset = datasets.MNIST(
        '/Users/liuzhengzuo/Desktop/PRML/Project 3_VAE/vae on own mac', train=False,
        transform=transforms.Compose([transforms.ToTensor(), torch.round]))

    # Load the test dataset
    x = torch.stack([test_dataset[i][0] for i in range(10000)]).reshape(-1, 784).to(device)
    id = test_dataset.targets

    # Convert id to one-hot encoding
    id = torch.zeros(10000, 10).scatter_(1, id.unsqueeze(1), 1)

    # Encode the subset of the test dataset
    if model.autoencoder_type == 'ae':
        z = model.encode(x)
    else:
        z = model.encode(x)[0]

    # Save the scattered image of the encoded latent vectors
    save_scattered_image(z.detach().cpu().numpy(), id, DIR, z_range, 'manifold.jpg')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Autoencoder MNIST Example')
    parser.add_argument(
        '--n_epochs', type=int, default=100,
        help="number of epochs (default: 100)")
    parser.add_argument(
        '--batch_size', type=int, default=128,
        help='batch size (default: 128)')
    parser.add_argument(
        '--lr', type=float, default=0.001,
        help='Learning rate for grad. descent (default: 0.001)')
    parser.add_argument(
        '--hidden_layer_size', type=int, default=512,
        help='hidden layer size (default: 512)')
    parser.add_argument(
        '--autoencoder_type', type=str, default='ae',
        help='Type of autoencoder: "ae" for autoencoder, "vae" for variational autoencoder (default: "ae")')
    parser.add_argument(
        '--exp_name', type=str, default='AE-arch=$autoencoder_type($hidden_layer_size)-lr=$lr-ep=$n_epochs')
    parser.add_argument(
       '--n_mc_samples', type=int, default=1,
       help='Number of Monte Carlo samples (default: 1)')
    parser.add_argument(
        '--seed', type=int, default=8675309,
        help='random seed (default: 8675309)')
    args = parser.parse_args()

    #  Create directory for saving results
    for key, val in args.__dict__.items():
        args.exp_name = args.exp_name.replace('$' + key, str(val))
    DIR = Path('/Users/liuzhengzuo/Desktop/PRML/Project 3_VAE/vae on own mac/results' + args.exp_name)
    DIR.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    device = torch.device("mps")

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            '/Users/liuzhengzuo/Desktop/PRML/Project 3_VAE/vae on own mac', train=True, download=True,
            transform=transforms.Compose([transforms.ToTensor(), torch.round])),
        batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            '/Users/liuzhengzuo/Desktop/PRML/Project 3_VAE/vae on own mac', train=False,
            transform=transforms.Compose([transforms.ToTensor(), torch.round])),
        batch_size=args.batch_size, shuffle=True)

    model = Autoencoder(hidden_layer_size=args.hidden_layer_size, autoencoder_type=args.autoencoder_type)
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.n_epochs):
        train_loss = train(epoch, model, optimizer, train_loader)
        with torch.no_grad():
            sample = torch.randn(64, 2).to(device)
            sample = model.decode(sample).cpu()
            save_image(sample.view(64, 1, 28, 28),
                       DIR / ('sample_' + str(epoch) + '.png'))

        test_loss = test(epoch, model, test_loader, DIR)
        print('  epoch %3d  test  loss %.3f' % (epoch, test_loss))

        row_df = pd.DataFrame([[epoch, train_loss, test_loss, args.lr]],
            columns=['epoch', 'train_loss', 'test_loss', 'lr'])
        csv_str = row_df.to_csv(
            None,
            float_format='%.8f',
            index=False,
            header=False if epoch > 0 else True,
            )
        if epoch == 0:
            with open(DIR / 'perf_metrics.csv', 'w') as f:
                f.write(csv_str)
        else:
            with open(DIR / 'perf_metrics.csv', 'a') as f:
                f.write(csv_str)

    torch.save(model.state_dict(), DIR / 'model.pt')
    plot_manifold_learning_result(model, DIR)





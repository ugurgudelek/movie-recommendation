__author__ = "Ugur Gudelek"
__email__ = "ugurgudelek@gmail.com"

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST


from utils import img_transform, save_image, to_img
from tsne import tsne_plot


import os

class AutoEncoder(nn.Module):


    def __init__(self, input_dim):
        super(AutoEncoder, self).__init__()

        self.input_dim = input_dim

        self.encoder = nn.Sequential(nn.Linear(self.input_dim, 20),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(20, 10))

        self.decoder = nn.Sequential(nn.Linear(10, 20),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(20, self.input_dim))

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def encode(self, x):
        self.train(False)
        latent_vector = self.encoder(x)
        self.train(True)
        return latent_vector

    def decode(self, latent_vector):
        self.train(False)
        reconstructed = self.decoder(latent_vector)
        self.train(True)
        return reconstructed










OUTPUT_PATH = '../output'
num_epochs = 100
batch_size = 128
learning_rate = 1e-3

os.makedirs(OUTPUT_PATH, exist_ok=True)

dataset = MNIST('../data', transform=img_transform, download=True)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

model = AutoEncoder(input_dim=28*28).cuda()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)


for epoch in range(num_epochs):
    for data in dataloader:
        img, label = data   # (batch_size, channel, width, height) -> (128, 1, 28, 28)
        img = img.view(img.size(0), -1)  # (batch_size, channel*width*height) -> (128, 784)
        img = img.cuda()
        # ===================forward=====================
        output = model(img)  # (128, 784)
        loss = criterion(output, img)
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # ===================log========================
    print('epoch [{}/{}], loss:{:.4f}'
          .format(epoch + 1, num_epochs, loss.item()))
    if epoch % 10 == 0:

        pic = to_img(img.cpu().data)
        save_image(pic, f'{OUTPUT_PATH}/input_image_{epoch}.png')

        pic = to_img(output.cpu().data)
        save_image(pic, f'{OUTPUT_PATH}/output_image_{epoch}.png')

        pic = to_img(model.decode(model.encode(img)).cpu().data)
        save_image(pic, f'{OUTPUT_PATH}/encode_decode_image_{epoch}.png')

        tsne_plot(X=img.cpu().numpy(), y=label.numpy(), filename=f'{OUTPUT_PATH}/tsne_input_{epoch}.html')
        tsne_plot(X=output.cpu().data.numpy(), y=label.numpy(), filename=f'{OUTPUT_PATH}/tsne_output_{epoch}.html')

torch.save(model.state_dict(), './sim_autoencoder.pth')















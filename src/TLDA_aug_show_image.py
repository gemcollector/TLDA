from __future__ import print_function
import warnings ; warnings.filterwarnings('ignore') # mute warnings, live dangerously ;)
from scipy.ndimage.filters import gaussian_filter
import torchvision

import os

import torch
import time
import torchgeometry as tgm



class K_MATRIX():

    def __init__(self, model, all_mask_pic, device, action_dim):
        self.device = torch.device("cuda:{}".format(device) if torch.cuda.is_available() else "cpu")
        self.all_mask_pic = torch.from_numpy(all_mask_pic).unsqueeze(0).float().to(self.device)
        self.gauss = tgm.image.GaussianBlur((13, 13), (3, 3))
        self.model = model
        self.action_dim = action_dim



    def score_frame(self, batch_state):
        all_attention_map = torch.zeros((batch_state.shape[0], batch_state.shape[2], batch_state.shape[3]))
        d = 7.5
        occlude = lambda I, mask: I * (1 - mask) + self.gauss(I.squeeze(1).float()).unsqueeze(1) * mask
        L = self.run_through_model(self.model, batch_state.to(self.device), interp_func=occlude)
        L = L.unsqueeze(1).repeat(1, self.all_mask_pic.shape[1], 1)
        l = self.run_through_model(self.model, batch_state.to(self.device), interp_func=occlude, mask=self.all_mask_pic)

        scores = torch.sum((L-l).pow(2), dim=2).mul_(.5).reshape((batch_state.shape[0], int(batch_state.shape[2] / d) + 1, int(batch_state.shape[3] / d) + 1))
        count = 0

        for pic in scores:
            pmax = pic.max()
            pic = torchvision.transforms.Resize(84)(pic.unsqueeze(0))
            attention_map = pmax * pic / pic.max()
            all_attention_map[count] = attention_map

            count += 1

        all_attention_map = all_attention_map.detach().cpu().data.numpy()

        return  all_attention_map


    def run_through_model(self, model, history, interp_func=None, mask=None):

        if mask is None:
            return model.actor.mlp(model.actor.encoder(history))
        else:
            history = history.unsqueeze(1)

            im = interp_func(history, mask)
            tens_state = im.reshape(history.shape[0]*self.all_mask_pic.shape[1], self.all_mask_pic.shape[2], self.all_mask_pic.shape[3], self.all_mask_pic.shape[4])

            return (model.actor.mlp(model.actor.encoder(tens_state))).reshape(history.shape[0], self.all_mask_pic.shape[1], self.action_dim)
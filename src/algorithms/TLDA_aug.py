from __future__ import print_function
import warnings ; warnings.filterwarnings('ignore') # mute warnings, live dangerously ;)


import torch
import torchgeometry as tgm

class K_MATRIX():

    def __init__(self, model, all_mask_pic, device, action_shape, args):
        self.device = torch.device("cuda:{}".format(device) if torch.cuda.is_available() else "cpu")
        self.all_mask_pic = torch.from_numpy(all_mask_pic).unsqueeze(0).float().to(self.device)
        self.width = args.width
        self.gauss = tgm.image.GaussianBlur((args.mean, args.mean), (args.std, args.std))
        self.model = model
        self.action_dim = action_shape


    def score_frame(self, batch_state):
        d = self.width
        occlude = lambda I, mask: I * (1 - mask) + self.gauss(I.squeeze(1).float()).unsqueeze(1) * mask
        with torch.inference_mode():
            L = self.run_through_model(self.model, batch_state.to(self.device), interp_func=occlude)
            L = L.unsqueeze(1).repeat(1, int(self.all_mask_pic.shape[1]), 1)
            init_l = self.run_through_model(self.model, batch_state.to(self.device), interp_func=occlude, mask=self.all_mask_pic)
            l = init_l
        scores = torch.sum((L-l).pow(2), dim=2).mul_(.5).reshape((batch_state.shape[0], int(batch_state.shape[2] / d), int(batch_state.shape[3] / d)))

        return  scores


    def run_through_model(self, model, history, interp_func=None, mask=None):
        if mask is None:
            return model.mlp(model.encoder(history))
        else:
            history = history.unsqueeze(1)
            im = interp_func(history, mask)
            tens_state = im.reshape(history.shape[0]*int(self.all_mask_pic.shape[1]), int(self.all_mask_pic.shape[2]), int(self.all_mask_pic.shape[3]), int(self.all_mask_pic.shape[4]))

            return (model.mlp(model.encoder(tens_state))).reshape(history.shape[0], int(self.all_mask_pic.shape[1]), self.action_dim)

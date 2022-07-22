import numpy as np
import torch
import torch.nn.functional as F
import utils
import augmentations
from algorithms.sac import SAC
from scipy.ndimage.filters import gaussian_filter
from algorithms.TLDA_aug import K_MATRIX




class TLDA(SAC):
	def __init__(self, obs_shape, action_shape, args):
		super().__init__(obs_shape, action_shape, args)
		self.lamba = args.lamba


		def get_mask(center, size, r):
			y, x = np.ogrid[-center[0]:size[0] - center[0], -center[1]:size[1] - center[1]]
			keep = x * x + y * y <= 1
			mask = np.zeros(size)
			mask[keep] = 1
			mask = gaussian_filter(mask, sigma=r)
			return mask / mask.max()

		def get_all_mask_pic(d=args.width, r=args.radius):
			cnt = 0
			all_mask_pic = np.zeros((int((np.ceil(obs_shape[1] / d)) ** 2), obs_shape[0], obs_shape[1], obs_shape[2]))
			for i in range(0, obs_shape[1], d):
				for j in range(0, obs_shape[2], d):
					mask = get_mask(center=[i, j], size=[obs_shape[1], obs_shape[2]], r=r)[np.newaxis, :]
					mask = np.concatenate([mask for i in range(obs_shape[0])])
					all_mask_pic[cnt, :, :] = mask
					cnt += 1

			return all_mask_pic

		all_mask_pic = get_all_mask_pic()
		self.tlda_model = K_MATRIX(self.actor, all_mask_pic, 0, action_shape[0]*2, args)


	def update_critic(self, obs, action, reward, next_obs, not_done, L=None, step=None):
		with torch.no_grad():
			_, policy_action, log_pi, _ = self.actor(next_obs)
			target_Q1, target_Q2 = self.critic_target(next_obs, policy_action)
			target_V = torch.min(target_Q1,
								 target_Q2) - self.alpha.detach() * log_pi
			target_Q = reward + (not_done * self.discount * target_V)

		if self.lamba == 1:
			obs_aug = (augmentations.random_conv(obs.clone()))

            # plus k_matrix
			k_matrix = self.tlda_model.score_frame(obs)
			tlda_aug = augmentations.choose_avg(obs, k_matrix, obs_aug)
			
			obs = utils.cat(obs, tlda_aug)
			action = utils.cat(action, action)
			target_Q = utils.cat(target_Q, target_Q)

			current_Q1, current_Q2 = self.critic(obs, action)
			critic_loss = (F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q))

		else:
			current_Q1, current_Q2 = self.critic(obs, action)
			critic_loss = (F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q))

			obs_aug = augmentations.random_conv(obs.clone())
            # plus k_matrix
			k_matrix, _ = self.tlda_model.score_frame(obs)
			tlda_aug = augmentations.choose_avg(obs, k_matrix, obs_aug)

			current_Q1_aug, current_Q2_aug = self.critic(tlda_aug, action)
			critic_loss += self.lamba * \
				(F.mse_loss(current_Q1_aug, target_Q) + F.mse_loss(current_Q2_aug, target_Q))

		if L is not None:
			L.log('train_critic/loss', critic_loss, step)
			
		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		self.critic_optimizer.step()

	def update(self, replay_buffer, L, step):
		obs, action, reward, next_obs, not_done = replay_buffer.sample_drq()

		self.update_critic(obs, action, reward, next_obs, not_done, L, step)

		if step % self.actor_update_freq == 0:
			self.update_actor_and_alpha(obs, L, step)

		if step % self.critic_target_update_freq == 0:
			self.soft_update_critic_target()

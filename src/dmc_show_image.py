import argparse

import numpy as np
import torch
import os
import utils
from video import VideoRecorder
import augmentations

from env.wrappers import make_env


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from TLDA_aug_show_image import K_MATRIX
from scipy.ndimage.filters import gaussian_filter
from algorithms.factory import make_agent

def parse_args():
    # environment
    parser = argparse.ArgumentParser()

    parser.add_argument('--domain_name', default='ball_in_cup')
    parser.add_argument('--task_name', default='catch')
    parser.add_argument('--frame_stack', default=3, type=int)
    parser.add_argument('--action_repeat', default=4, type=int)
    parser.add_argument('--episode_length', default=1000, type=int)
    parser.add_argument('--eval_mode', default='train', type=str)

    # agent
    parser.add_argument('--algorithm', default='drq', type=str)
    parser.add_argument('--train_steps', default='500k', type=str)
    parser.add_argument('--discount', default=0.99, type=float)
    parser.add_argument('--init_steps', default=1000, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--hidden_dim', default=1024, type=int)

    # actor
    parser.add_argument('--actor_lr', default=1e-3, type=float)
    parser.add_argument('--actor_beta', default=0.9, type=float)
    parser.add_argument('--actor_log_std_min', default=-10, type=float)
    parser.add_argument('--actor_log_std_max', default=2, type=float)
    parser.add_argument('--actor_update_freq', default=2, type=int)

    # critic
    parser.add_argument('--critic_lr', default=1e-3, type=float)
    parser.add_argument('--critic_beta', default=0.9, type=float)
    parser.add_argument('--critic_tau', default=0.01, type=float)
    parser.add_argument('--critic_target_update_freq', default=2, type=int)

    # architecture
    parser.add_argument('--num_shared_layers', default=4, type=int)
    parser.add_argument('--num_head_layers', default=0, type=int)
    parser.add_argument('--num_filters', default=32, type=int)
    parser.add_argument('--projection_dim', default=50, type=int)
    parser.add_argument('--encoder_tau', default=0.05, type=float)

    # entropy maximization
    parser.add_argument('--init_temperature', default=0.1, type=float)
    parser.add_argument('--alpha_lr', default=1e-4, type=float)
    parser.add_argument('--alpha_beta', default=0.5, type=float)

    # svea
    parser.add_argument('--svea_alpha', default=0.5, type=float)
    parser.add_argument('--svea_beta', default=0.5, type=float)

    # eval
    parser.add_argument('--save_freq', default='100k', type=str)
    parser.add_argument('--eval_freq', default='10k', type=str)
    parser.add_argument('--eval_episodes', default=30, type=int)
    parser.add_argument('--distracting_cs_intensity', default=0., type=float)

    # misc
    parser.add_argument('--seed', default=3, type=int)
    parser.add_argument('--log_dir', default='logs', type=str)
    parser.add_argument('--save_video', default=False, action='store_true')

    args = parser.parse_args()

    args.image_size = 84
    args.image_crop_size = 84

    return args





def main(args):
    # Initialize environment
    utils.set_seed_everywhere(args.seed)
    env = make_env(
        domain_name=args.domain_name,
        task_name=args.task_name,
        seed=args.seed+42,
        episode_length=args.episode_length,
        action_repeat=args.action_repeat,
        image_size=args.image_size,
        mode='train'
    )
    args.work_dir = '/home/yzc/dmcontrol-generalization-benchmark/logs/{}_{}/{}/{}/model'.format(args.domain_name, args.task_name, args.algorithm, args.seed)
    model_dir = utils.make_dir(args.work_dir)

    cropped_obs_shape = (3 * args.frame_stack, 84, 84)
    agent = make_agent(
        obs_shape=cropped_obs_shape,
        action_shape=env.action_space.shape,
        args=args
    )

    def get_mask(center, size, r):
        y, x = np.ogrid[-center[0]:size[0] - center[0], -center[1]:size[1] - center[1]]
        keep = x * x + y * y <= 1
        mask = np.zeros(size)
        mask[keep] = 1
        mask = gaussian_filter(mask, sigma=r)
        return mask / mask.max()

    def get_all_mask_pic(d=5, r=5):
        cnt = 0
        all_mask_pic = np.zeros((int((np.ceil(args.image_size / d)) ** 2), cropped_obs_shape[0], cropped_obs_shape[1], cropped_obs_shape[2]))
        for i in range(0, args.image_size, d):
            for j in range(0, args.image_size, d):
                mask = get_mask(center=[i, j], size=[args.image_size, args.image_size], r=r)[np.newaxis, :]
                mask = np.concatenate([mask for i in range(cropped_obs_shape[0])])
                all_mask_pic[cnt, :, :] = mask
                cnt += 1

        return all_mask_pic

    all_mask_pic = get_all_mask_pic()


    step=500000
    agent.actor.load_state_dict(torch.load('%s/actor_%s.pt' % (model_dir, step), map_location='cpu'))
    aim_model = K_MATRIX(agent, all_mask_pic, 0, env.action_space.shape[0]*2)


    args.save_video = True
    video_dir = utils.make_dir(os.path.join(args.work_dir, 'video'))
    video = VideoRecorder(video_dir if args.save_video else None)


    import imageio
    tlda_obs_images = []

    def evaluate(env, agent, video, num_episodes, step):
        """Evaluate agent"""
        for i in range(num_episodes):
            obs = env.reset()
            video.init(enabled=(i == 0))
            done = False
            episode_reward = 0
            while not done:
                torch_obs = torch.from_numpy(np.array(obs)).unsqueeze(0)
                aim_attention = aim_model.score_frame(torch_obs)

                # aug_obs = augmentations.random_grayscale(torch_obs.clone().cuda())[0][0].unsqueeze(0)
                aug_obs = augmentations.random_pepper(torch_obs.clone().cuda(), 0.8)

                tlda_obs = augmentations.choose_avg(torch_obs.cuda(), torch.from_numpy(aim_attention), aug_obs.cuda())
                with utils.eval_mode(agent):
                    action = agent.sample_action(tlda_obs[0].cpu().data.numpy())
                    # action = agent.sample_action(aug_obs[0].cpu().data.numpy())

                tlda_obs_images.append(tlda_obs[0][6:9].cpu().data.numpy().transpose(1, 2, 0))

                obs, reward, done, _ = env.step(action)
                video.record(env)
                episode_reward += reward

            video.save('%d.mp4' % step)
            imageio.mimsave('tlda_obs_{}_{}.gif'.format(args.domain_name, args.task_name),
                            [np.array(img) for i, img in enumerate(tlda_obs_images) if i % 1 == 0], fps=15)

            print('Reward:', episode_reward)


    evaluate(env, agent, video, 1, 999)




if __name__ == '__main__':
    args = parse_args()
    main(args)
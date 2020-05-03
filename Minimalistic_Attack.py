import os
import cv2
import numpy as np
from IPython import display
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import time
import gym
from pybrain.optimization import GA, CMAES, HillClimber, RandomSearch
from stable_baselines import DQN, PPO2, A2C, ACER, ACKTR, SAC
from Utils.model_loader import model_loader
from Utils.envGym import envGym
from stable_baselines.common.atari_wrappers import make_atari
from stable_baselines.deepq.policies import MlpPolicy, CnnPolicy
from stable_baselines.common.cmd_util import make_atari_env
from stable_baselines.common import make_vec_env
from stable_baselines.common.vec_env import VecFrameStack, DummyVecEnv
from cv2 import *


from multiprocessing import Pool
# import cv2
# from multiprocessing import Pool
# from itertools import product
from functools import partial

# import os
# mute os log
os.environ['KMP_WARNINGS'] = 'off'

def get_path(game, home_dir='./SAC_model'):
    models = {
        'Qbert':          ['Qbert', '24', '6', True], 
        'Assault':        ['Assault',	'20','6', False],
        'Enduro':         ['Enduro',	'12', '3', True	],
        'BeamRider':      ['BeamRider', '24',	 '3', False],
        'SpaceInvaders':  ['SpaceInvaders', '20',	'6',False],
        'Breakout':       ['Breakout', '6', '6', True]}
    
    conf = models[game]
    model_dir = f'{home_dir}/saved_models/sac_discrete_atari_{conf[0]}-v4/sac_discrete_atari_{conf[0]}-v4_s{conf[2]}/'
    model_save_name = f'tf1_save{conf[1]}'
    return model_dir + model_save_name

def main(game, method, pixels, tca, runname, deterministic, customized=False, run=1):
    
    def obj(variable, actions_0, obs):
        brightness = np.min([obs.max(), 254])  # this takes min of the max brightness of the obs and 254, original code put 254
        perturbation = np.zeros_like(obs)
        for i in range(len(variable) // 3):
            x = int(np.round(variable[3*i]*83))
            y = int(np.round(variable[3*i+1]*83))
            pixel_attack = int(np.round(variable[3*i+2]*brightness))
            x = np.clip(x, 0, 83)
            y = np.clip(y, 0, 83)
            pixel_attack = np.clip(pixel_attack, 0, brightness)
            if len(obs.shape) < 4:
                perturbation[ x, y, :] = pixel_attack 
            else:
                perturbation[:, x, y, :] = pixel_attack 
        np.clip(perturbation, 0, brightness)
        obs_new = obs + perturbation
        obs_new[obs_new > brightness] = brightness  # restrict the max intensity to 254
        obs_new[obs_new < 0] = 0
        actions_new = model.action_probability(obs_new)
        if len(obs.shape) < 4:
            fitness_value = max_max_distance(np.expand_dims(actions_new, axis=0), np.expand_dims(actions_0, axis=0))
        else:
            fitness_value = max_max_distance(actions_new, actions_0)
        return fitness_value

    def evaluate(variable, obs):
        brightness = np.min([obs.max(), 254])
        perturbation = np.zeros_like(obs)
        for i in range(len(variable) // 3):
            x = int(np.round(variable[3 * i] * 83))
            y = int(np.round(variable[3 * i + 1] * 83))
            pixel_attack = int(np.round(variable[3 * i + 2] * brightness))
            x = np.clip(x, 0, 83)
            y = np.clip(y, 0, 83)
            pixel_attack = np.clip(pixel_attack, 0, brightness)
            if len(obs.shape)<4:
                perturbation[x, y, :] = pixel_attack
            else:
                perturbation[: , x, y, :] = pixel_attack
            # print('pixel_attack', x, y, pixel_attack)  
        np.clip(perturbation, 0, brightness)
        obs_new = obs + perturbation
        obs_new[obs_new > brightness] = brightness
        obs_new[obs_new < 0] = 0
        actions = model.action_probability(obs)
        actions_new = model.action_probability(obs_new)
        if len(obs.shape) < 4:
            action = model.predict(obs_new)
        else:
            action, _states = model.predict(obs_new)

        obs_candi, rewards, dones, infos = env.step(action)
        return obs_candi, rewards, dones, infos, obs_new, actions_new, perturbation

    def minmax_distance(actions_new, actions_0):
        arg_max = np.argmax(actions_0[0])
        arg_min = np.argmin(actions_0[0])
        minmax_dist = actions_new[0][arg_min] - actions_new[0][arg_max]
        return minmax_dist
    
    def max_max_distance(actions_new, actions_0):
        arg_max = np.argmax(actions_0[0])
        a_candid = list(actions_new[0])
        a_candid.remove(a_candid[arg_max])
        maxmax_dist = np.max(a_candid) - actions_new[0][arg_max]
        return maxmax_dist
    
    def calculate_entropy(actions):
        entropy_actions = [-probs * np.log(probs) / np.log(len(actions)) if probs>0 else 0 for probs in actions]
        entropy = np.sum(entropy_actions)
        return entropy

    alg       = GA

    # note: choose a model to run
    modelMap = {
        'dqn'  : DQN,
        'a2c'  : A2C,
        'ppo2' : PPO2,
        'acktr': ACKTR,
        'sac': SAC
    }

    if customized:
        customized_path = get_path(game = game)
        dirs = customized_path.split("/")
        model_save_name = dirs[-1]
        model_dir = "/".join(dirs[:-1])
        model = model_loader(model_dir, model_save_name)
        print(f"Model {customized_path} loaded")
    elif '.pkl' in game or '.pickle' in game:
        print("enter new load")
        model = modelMap[method].load("trained_agents/{}/{}".format(method, game))
        print(f"Model {method} loaded")
    else:
        print("enter original load")
        model = modelMap[method].load("trained_agents/{}/{}NoFrameskip-v4.pkl".format(method, game))
        print(f"Model {method}NoFrameskip-v4 loaded")


    if customized:
        original_env = gym.make(model.config['rl_params']['env_name'])
        env = envGym(original_env, 4)
        obs = env.reset()
        model.action_probability = model.get_action_probabilistic

        model.predict = lambda obs: model.get_action(obs, deterministic=deterministic)
        # env.render()

    elif '.pkl' in game or '.pickle' in game:

        # env = make_vec_env("LunarLanderContinuous-v2", n_envs=20)
        env = make_vec_env(game.split(".")[0], n_envs=20)
        
    else:
        env = make_atari_env('{}NoFrameskip-v4'.format(game), num_env=1, seed=run)
        print(f"Make environment {game}NoFrameskip-v4: Done")
        env = VecFrameStack(env, n_stack=4)
        env.reset()
        model.set_env(env)
        obs = env.reset()
   

    Episode_Reward = []
    Episode_Lenth  = []
    Attack_times   = []
    dir_name = 'results/{}/{}_{}/{}/FSA_{}_TCA_{}'.format(runname, method, deterministic, game, pixels, tca)
    print(f"result will be save in {dir_name}")
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    atk_num = pixels
    bounds = [[0,1] ,[0,1], [0,1]] * atk_num


   
    x0 = [0.5, 0.5, 0.5] * atk_num
    atk_time = 0
    TrueS_array = []
    Delta_array = []
    CleanS_array = []

    print(f"Start training! run number {run}")
    epoch = 10000
    acc_solution = []

    for i in range(epoch):

        if i % 10 == 0:              # cache our model every <save_epoch_freq> epochs
            print(f'Training model for run {run} epoch {i} / {epoch}')
        actions = model.action_probability(obs)

        if customized:
            attack_significance = calculate_entropy(actions)
            CleanS_array.append((obs[:, :, 3]).astype('uint8'))
        else:
            attack_significance = calculate_entropy(actions[0])
            CleanS_array.append((obs[0, :, :, 3]).astype('uint8'))
  
        if attack_significance <= tca:
            atk_time = atk_time+1
            l = alg(lambda variable: obj(variable, actions, obs), x0, xBound=bounds, verbose=False)
            l.maximize = True
            l.maxEvaluations = 400
            res = l.learn()
            solution = list(res)[0]
            acc_solution.append(solution)
            obs, rewards, dones, infos, obs_new, actions_new, perturbation = evaluate(solution, obs)
            obs_store = np.int_(obs_new)
            # print(f"min obs {obs.min()}, max obs {obs.max()}")
            # print(f"min obs_new {obs_new.min()}, max obs {obs_new.max()}")
            if customized:
                true_state = (obs_store[:, :, 3]).astype('uint8')
                Delta_array.append(perturbation[:, :, 3].astype('uint8'))
            else:
                true_state = (obs_store[0, :, :, 3]).astype('uint8')
                Delta_array.append(perturbation[0, :, :, 3].astype('uint8'))

            # if true_state.min()<0:
            #     print("min smaller than 0")
            #     pass
            TrueS_array.append(true_state)
        else:
            obs = np.int_(obs)
            # print(f"min obs {obs.min()}, max obs {obs.max()}")
            if customized:
                true_state = (obs[:, :, 3]).astype('uint8')
            else:
                true_state = (obs[0, :, :, 3]).astype('uint8')
            TrueS_array.append(true_state)
            Delta_array.append(np.zeros([84, 84]).astype('uint8'))
            # if true_state.min()<0:
            #     print("min smaller than 0")
            #     pass

            if customized:
                action = model.predict(obs)
            else: 
                action, _states = model.predict(obs)
            
            obs, rewards, dones, infos = env.step(action)

        if customized:
            episode_infos = infos.get('episode')
        else:
            episode_infos = infos[0].get('episode')

        if episode_infos is not None:
            print("Atari Episode Score: {:.2f}".format(episode_infos['r']))
            print("Atari Episode Length", episode_infos['l'])
            REWARD = episode_infos['r']  # length of REWARD
            Lenth = episode_infos['l']  # length of episode
            break

        if dones:
            pass

    size = (84, 84)
    video_dir = 'results/{}_videos/{}_{}/{}/FSA_{}_TCA_{}'.format(runname, method, deterministic, game, pixels, tca)
    if not os.path.exists(video_dir):
        os.makedirs(video_dir)
    fps = 10
    out_true  = cv2.VideoWriter('{}/true_run_{}.avi'.format(video_dir, run), cv2.VideoWriter_fourcc(*'XVID'), fps, size)  #*'PIM1'
    out_delta = cv2.VideoWriter('{}/delta_run_{}.avi'.format(video_dir, run), cv2.VideoWriter_fourcc(*'XVID'), fps, size)
    out_clean = cv2.VideoWriter('{}/clean_run_{}.avi'.format(video_dir, run), cv2.VideoWriter_fourcc(*'XVID'), fps, size)



    for i in range(len(TrueS_array)):
        image_true = TrueS_array[i]
        x_true = cv2.cvtColor(image_true, cv2.COLOR_GRAY2RGB)
        # x_true = np.repeat(image_true, 3, axis=1)
        # x_true = x_true.reshape(84, 84, 3)
        # x_true[:, :, 0] = 150 * np.ones((84, 84), dtype=int)
        # x_true[:, :, 1] = 150 * np.ones((84, 84), dtype=int)
        out_true.write(x_true)
        image_delta = Delta_array[i]
        x_delta = cv2.cvtColor(image_delta, cv2.COLOR_GRAY2RGB)
        # x_delta = np.repeat(image_delta, 3, axis=1)
        # x_delta = x_delta.reshape(84, 84, 3)
        # x_delta[:, :, 0] = 150 * np.ones((84, 84), dtype=int)
        # x_delta[:, :, 1] = 150 * np.ones((84, 84), dtype=int)
        out_delta.write(x_delta)
        image_clean = CleanS_array[i]
        x_clean = cv2.cvtColor(image_clean, cv2.COLOR_GRAY2RGB)
        # x_clean = np.repeat(image_clean, 3, axis=1)
        # x_clean = x_clean.reshape(84, 84, 3)
        # x_clean[:, :, 0] = 150 * np.ones((84, 84), dtype=int)
        # x_clean[:, :, 1] = 150 * np.ones((84, 84), dtype=int)
        out_clean.write(x_clean)
    cv2.destroyAllWindows()
    out_true.release()
    out_delta.release()
    out_clean.release()
    Episode_Reward.append(REWARD)
    Episode_Lenth.append(Lenth)
    Attack_times.append(atk_time)
    np.array([REWARD,atk_time, Lenth])
    # data = np.column_stack(Episode_Reward, Attack_times, Episode_Lenth)
    np.savetxt(f"{dir_name}/run_{run}_attacks.csv", np.array(acc_solution), delimiter=",")
    np.savetxt('{}/run_{}.csv'.format(dir_name, run), np.array([REWARD, atk_time, Lenth]), delimiter=",")

def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('-g', '--game',
                        help="the atari game",
                        default='Breakout', type=str)
    parser.add_argument('-a', '--algorithm',
                        help="the algorithm for the policy training ",
                        default='dqn', type=str)
    parser.add_argument('-n', '--pixels',
                        help="The number of pixels",
                        default=5, type=int)
    parser.add_argument('-t', '--tca',
                        help="The TCA boundary",
                        default=0.9, type=float)
    parser.add_argument('-r', '--runname',
                        help="The run name",
                        default='test', type=str)
    parser.add_argument('--use_gpu', action='store_true', default=False)
    parser.add_argument('--gpu', '-gpu', default='0,1', type=str)
    parser.add_argument('--customized', action='store_true', default=False) 
    parser.add_argument('--d', action='store_true', default=False)    
    args = parser.parse_args()
    return args.game, args.algorithm, args.pixels, args.tca, args.runname, args.customized, args.use_gpu, args.gpu, args.d

if __name__ == '__main__':
    # note: You could change pool number and X_input at the same time
    p = Pool(5)
    
    game, method, pixels, tca, runname,  customized, use_gpu, gpu, deterministic = parse_arguments()


    # Tensorflow    
    if(use_gpu):
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu
        # configure gpu use and supress tensorflow warnings
        import tensorflow as tf
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
        print('Tensorflow imported, version',tf.__version__)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.6
        print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
        
    
    # X_input = list(range(1,5))
    X_input = list(range(1, 6))
    # main(game, method, pixels, tca, runname, deterministic, customized, 1)
    func = partial(main, game, method, pixels, tca, runname, deterministic, customized)
    p.map(func,X_input)
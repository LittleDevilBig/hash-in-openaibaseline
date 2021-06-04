import sys
import multiprocessing 
import os
import os.path as osp
import gym
from collections import defaultdict
import tensorflow as tf
import numpy as np

import random

from baselines.common.vec_env.vec_frame_stack import VecFrameStack
from baselines.common.cmd_util import common_arg_parser, parse_unknown_args, make_mujoco_env, make_atari_env
from baselines.common.tf_util import save_state, load_state, get_session
from baselines import bench, logger
from importlib import import_module

from baselines.common.vec_env.vec_normalize import VecNormalize
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common import atari_wrappers, retro_wrappers
from baselines.common.models import process_action
from baselines.ha2c.Config import config
import time
import cv2
from baselines.common.running_mean_std import RunningMeanStd

import matplotlib.pyplot as plt

import pylab

try:
    from mpi4py import MPI
except ImportError:
    MPI = None


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

_game_envs = defaultdict(set)
for env in gym.envs.registry.all():
    # solve this with regexes
    env_type = env._entry_point.split(':')[0].split('.')[-1]
    _game_envs[env_type].add(env.id)
cwd_dir = os.getcwd()
sys.path.append(cwd_dir+'/ha2c/ha2c.py')

# reading benchmark names directly from retro requires 
# importing retro here, and for some reason that crashes tensorflow 
# in ubuntu 
_game_envs['retro'] = set([
    'BubbleBobble-Nes',
    'SuperMarioBros-Nes',
    'TwinBee3PokoPokoDaimaou-Nes',
    'SpaceHarrier-Nes',
    'SonicTheHedgehog-Genesis',
    'Vectorman-Genesis',
    'FinalFight-Snes',
    'SpaceInvaders-Snes',
])


def train(args, extra_args):


    env_type, env_id = get_env_type(args.env)
        
    total_timesteps = int(args.num_timesteps)
    seed = args.seed

    if args.save_path is not None:  
       extra_args['save_path']=args.save_path


    learn = get_learn_function(args.alg)
    alg_kwargs = get_learn_function_defaults(args.alg, env_type)
    alg_kwargs.update(extra_args)

    env = build_env(args)

    if args.network:
        alg_kwargs['network'] = args.network
    else:
        if alg_kwargs.get('network') is None:
            alg_kwargs['network'] = get_default_network(env_type)
 
       
    
    print('Training {} on {}:{} with arguments \n{}'.format(args.alg, env_type, env_id, alg_kwargs))

    if args.alg=='ha2c' or args.alg=='arnd':
        model = learn(
            env=env,  
            seed=seed,
            total_timesteps=total_timesteps,
            args=args,
            **alg_kwargs
        )
    else:
        model = learn(
            env=env,  
            seed=seed,
            total_timesteps=total_timesteps,
            **alg_kwargs
        )

    return model, env


def build_env(args, log_prefix=''):
    ncpu = multiprocessing.cpu_count()
    if sys.platform == 'darwin': ncpu //= 2
    nenv = args.num_env or ncpu
    alg = args.alg
    rank = MPI.COMM_WORLD.Get_rank() if MPI else 0
    seed = args.seed    

    env_type, env_id = get_env_type(args.env)
    if env_type == 'mujoco':
        get_session(tf.ConfigProto(allow_soft_placement=True,
                                   intra_op_parallelism_threads=1, 
                                   inter_op_parallelism_threads=1))

        if args.num_env:
            env = SubprocVecEnv([lambda: make_mujoco_env(env_id, seed + i if seed is not None else None, args.reward_scale) for i in range(args.num_env)])    
        else:
            env = DummyVecEnv([lambda: make_mujoco_env(env_id, seed, args.reward_scale)])

        env = VecNormalize(env)

    elif env_type == 'atari':
        if alg == 'acer':
            env = make_atari_env(env_id, nenv, seed)
        elif alg == 'deepq':
            env = atari_wrappers.make_atari(env_id)
            env.seed(seed)
            env = bench.Monitor(env, logger.get_dir())
            env = atari_wrappers.wrap_deepmind(env, frame_stack=True, scale=True)
        elif alg == 'trpo_mpi':
            env = atari_wrappers.make_atari(env_id)
            env.seed(seed)
            env = bench.Monitor(env, logger.get_dir() and osp.join(logger.get_dir(), str(rank)))
            env = atari_wrappers.wrap_deepmind(env)
            # TODO check if the second seeding is necessary, and eventually remove
            env.seed(seed)
        else:
            frame_stack_size = 4
            env = VecFrameStack(make_atari_env(env_id, nenv, seed, log_prefix=log_prefix), frame_stack_size)

    elif env_type == 'retro':
        import retro
        gamestate = args.gamestate or 'Level1-1'
        env = retro_wrappers.make_retro(game=args.env, state=gamestate, max_episode_steps=10000, use_restricted_actions=retro.Actions.DISCRETE)
        env.seed(args.seed)
        env = bench.Monitor(env, logger.get_dir())
        env = retro_wrappers.wrap_deepmind_retro(env)
        
    elif env_type == 'classic_control':
        def make_env():
            e = gym.make(env_id)
            e = bench.Monitor(e, logger.get_dir(), allow_early_resets=True)
            e.seed(seed)
            return e
            
        env = DummyVecEnv([make_env])

    else:
        raise ValueError('Unknown env_type {}'.format(env_type))

    return env


def get_env_type(env_id):
    if env_id in _game_envs.keys():
        env_type = env_id
        env_id =  [g for g in _game_envs[env_type]][0]
    else:
        env_type = None
        for g, e in _game_envs.items():
            if env_id in e:
                env_type = g
                break 
        assert env_type is not None, 'env_id {} is not recognized in env types'.format(env_id, _game_envs.keys())

    return env_type, env_id

def get_default_network(env_type):
    if env_type == 'mujoco' or env_type == 'classic_control':
        return 'mlp'
    if env_type == 'atari':
        return 'cnn'

    raise ValueError('Unknown env_type {}'.format(env_type))
    
def get_alg_module(alg, submodule=None):
    submodule = submodule or alg
    try:
        # first try to import the alg module from baselines
        alg_module = import_module('.'.join(['baselines', alg, submodule]))
    except ImportError:
        # then from rl_algs
        alg_module = import_module('.'.join(['rl_' + 'algs', alg, submodule]))
    
    return alg_module
        

def get_learn_function(alg):
    return get_alg_module(alg).learn

def get_learn_function_defaults(alg, env_type):
    try:
        alg_defaults = get_alg_module(alg, 'defaults')
        kwargs = getattr(alg_defaults, env_type)()
    except (ImportError, AttributeError):
        kwargs = {}       
    return kwargs
    
def parse(v): 
    '''
    convert value of a command-line arg to a python object if possible, othewise, keep as string
    '''

    assert isinstance(v, str)
    try:
        return eval(v) 
    except (NameError, SyntaxError): 
        return v


def set_global_seeds(i):
    try:
        import tensorflow as tf
    except ImportError:
        pass
    else:
        from mpi4py import MPI
        tf.set_random_seed(i)
    np.random.seed(i)
    random.seed(i)


def position_encoding_init(n_position_,d_pos_vec, coef=1):
    position_enc = np.array([
         [pos / np.power(10000, 2*i/d_pos_vec) for i in range(d_pos_vec)] 
         if pos !=0 else np.zeros(d_pos_vec) for pos in range(n_position_)
        ])

    #for j is even
    position_enc[1:, 0::2] = coef*np.sin(position_enc[1:,0::2])
    #for j is odd
    position_enc[1:, 1::2] = coef*np.cos(position_enc[1:,1::2])

    return position_enc

def main():
    # configure logger, disable logging in child MPI processes (with rank > 0) 
              
    arg_parser = common_arg_parser()
    args, unknown_args = arg_parser.parse_known_args()
    extra_args = {k: parse(v) for k,v in parse_unknown_args(unknown_args).items()}

    
    if MPI is None or MPI.COMM_WORLD.Get_rank() == 0:
        rank = 0
        logger.configure(dir=args.log_dir)
    else:
        logger.configure(dir=args.log_dir,format_strs = [])
        rank = MPI.COMM_WORLD.Get_rank()

    #seed = MPI.COMM_WORLD.Get_rank()
    #set_global_seeds(seed)



    model, _ = train(args, extra_args)

    if args.save_path is not None and rank == 0:
        save_path = osp.expanduser(args.save_path)
        model.save(save_path)
    

    max_count=3

    goal_max_life = 5
    life_count = 0

    if args.play:
        logger.log("Running trained model")
        env = build_env(args)
        last_obs = obs = env.reset()

        obs_g = obs

        dones = [False for _ in range(env.num_envs)]
        states = model.initial_state


        flag= False

        a=0.1
        mean = 0 
        count =0

        sess = get_session()
        if args.alg=='ha2c':

            xr_states = model.xr_initial_state
            xrp_states = model.xrp_initial_state

            obs_mean,obs_var = model.get_mean_var()
            pv_mean, pv_var, epv_mean, epv_var = model.get_rms_var()
            exp_reward_rms = RunningMeanStd(shape=())


            if config['pe_matrix'] and config['PE_type']=='normal':
                PE = model.get_norm_PE()
            else:
                PE = position_encoding_init(64,100, coef=2.5)

            state_list=[]

        if args.alg =='arnd':
             xr_states = model.xr_initial_state
             xrp_states = model.xrp_initial_state
             obs_mean,obs_var = model.get_mean_var()

        unwrapped_env = gym.make('SeaquestNoFrameskip-v4').unwrapped
        while True:


            for i in range(5):


                if args.alg=='ha2c':
                    last_obs = obs
                    last_dones= dones
                    last_state = states
    


                    if config['use_q']:
                        actions, states = model.q_step(obs, S=states, M=dones)
                        values = np.zeros_like(actions)
                        exp_actions = values
                        exp_v = values
                    else:
                        actions, values, exp_v, states, aprobs= model.step(obs, S=states, M=dones)
                    #if count %10 ==0 or count %10 ==1 or count %10 ==2 or count %10 ==3 or count %10 ==4:
                    #    actions = exp_actions



                    a = np.random.uniform() < 0.5
                    a = False
                    #if a:
                    #    actions = exp_actions



                    #exp_rewards, target_h = model.gen_expr(obs, X_mean=obs_mean, X_var=obs_var)



                elif args.alg=='arnd':
                    actions, values, int_v, states, aprobs= model.step(last_obs, S=states, M=dones)
                else:
                    actions, values, states, _= model.step(obs, S=states, M=dones)
                #pred_action_prob, pred_a_latent = model.pred_a(last_obs,obs)

                src_actions = np.copy(actions)
                t=''
                t = input("input:")
                if t!='':
                    t=int(t)
                    if t<=17:
                        actions=[t]

                #states = env.clone_full_state()
                print(actions)
                #state_list.append(states)
                obs, rewards, dones, infos  = env.step(actions)
    

                #print(infos)
                #print(np.linalg.norm(vf_latent), values,rewards)
                #l2 = np.linalg.norm(obs-last_obs)
                #if mean ==0:
                #    mean = l2
                #else:
                #    mean = 0.9*mean + 0.1*l2
                #print(l2, mean, l2/mean)
                #a = np.asarray(obs)
                #cv2.imshow('', a)
                #prob_rinv, prob_dinv, prob_rfor, prob_dfor, f_bonuses, dists, count_prob \
                #                                = model.get_all_probe(last_obs,actions,obs,last_dones, S=last_state, S_next= states, M=last_dones, M_next= dones,\
                #                                                             S_inv=last_state_inv ,S_for=last_state_for, M_inv=last_dones, M_for=last_dones)
    
                reward = rewards.any()
                done = dones.any() if isinstance(dones, np.ndarray) else dones
                #print("d:",prob_dfor[0][1],done, "r:",prob_rfor[0][0], reward,"value:",values)
                #logger.log("shape:",obs.shape)
    

               
                #print(img)
                #cv2.imshow('', img)
    

                time.sleep(0.04)


                count = count +1
                if done:
                    mean = 0 
                    count = 0

                    if 'episode' in infos[0]:
                       logger.log("ep_r:{},ep_len:{}".format(infos[0]['episode']['r'],int(infos[0]['episode']['l'])))
                    obs = env.reset()
                

                if args.alg=='ha2c':
                    norm_obs = np.clip((obs - obs_mean)/ np.sqrt(obs_var),-5,5)
    
                    '''
                    print(np.max(norm_obs))
                    print(np.mean(norm_obs))
                    print(np.min(norm_obs))
                    zeros_obs = np.zeros_like(obs)
                    '''
                    if config['pyramid_RND']:
                        local_exp_rewards, global_exp_rewards = model.get_exp_bonus(last_obs, actions,obs, log=True, X_mean=obs_mean, X_var=obs_var, RND_XR_S=xr_states, RND_XRP_S=xrp_states, RND_XR_M=dones, RND_XRP_M=dones)
                        
                        exp_rewards = local_exp_rewards + global_exp_rewards
    
                        print("exp_v: ", exp_v, "v: ", values, "all: ", exp_v+values)
    
                    elif config['use_matrix_RND']:
    
                        if config['super_matrix'] or config['ss_matrix'] or config['diff_RND'] or config['pe_matrix']:
                            local_exp_rewards, _, global_exp_rewards = model.get_exp_bonus(last_obs, actions, obs,log=True, X_mean=obs_mean, X_var=obs_var, \
                                                                RND_XR_S=xr_states, RND_XRP_S=xrp_states, RND_XR_M=dones, RND_XRP_M=dones)
                            
                            exp_rewards = local_exp_rewards + global_exp_rewards
                            #print(int_rew)
                        else:
                            int_rew1, int_rew2, int_rew3, int_rew4, int_rew5, int_rew = model.get_exp_bonus(last_obs, actions,obs, X_mean=obs_mean, X_var=obs_var, RND_XR_S=xr_states, RND_XRP_S=xrp_states, RND_XR_M=dones, RND_XRP_M=dones)
                            exp_rew = int_rew1 + int_rew2 + int_rew3 + int_rew4 
                            print(int_rew1, int_rew2, int_rew3, int_rew4, int_rew5, int_rew)
        
                            exp_rewards = np.maximum(int_rew1, int_rew2)
                            exp_rewards = np.maximum(exp_rewards, int_rew3)
                            exp_rewards = np.maximum(exp_rewards, int_rew4)
                            exp_rewards = np.maximum(exp_rewards, int_rew5)
                            exp_rewards = exp_rewards + int_rew
    
                        xr_states=None
                        xrp_states=None
    
                        print("exp_v: ", exp_v, "v: ", values, "all: ", exp_v+values)
                    else:
                        exp_rewards,xr_states,xrp_states = model.get_exp_bonus(last_obs, actions, obs, X_mean=obs_mean, X_var=obs_var, RND_XR_S=xr_states, RND_XRP_S=xrp_states, RND_XR_M=dones, RND_XRP_M=dones)
                        
                        if config['use_forward_model']:
                            forward_bonus = model.get_forward_bonus(last_obs, obs, actions, X_mean=obs_mean, X_var=obs_var)
                        else:
                            forward_bonus = 0
    
    
                        print("exp_rewards:",exp_rewards, forward_bonus, exp_v, values, done, exp_v+values)
                        #print("norm_v: ", norm_v, "norm_expv:", norm_expv, "sum:", norm_v+norm_expv)
                        #print(pv_mean, pv_var)
                        
                    #target_select_c, target_logits = model.get_target_count(obs, actions, X_mean=obs_mean, X_var=obs_var)
    
                    '''
                    norm_v =  (values  - pv_mean) / np.sqrt(pv_var)
                    norm_expv = (exp_v  - epv_mean) / np.sqrt(epv_var)
    
                    mapping_v = norm_v * np.sqrt(epv_var) + epv_mean
                    '''
    
                    N = exp_rewards + 1
                    N_i = np.zeros_like(N)
                    for i in range(env.num_envs):
                        N_i[i] =  aprobs[i,actions[i]] #* N[i] + 0.001
        
                    exp_rewards = N /  N_i
                    print(aprobs)
                    print(exp_rewards)
                    
                    #print("mapping_v:",mapping_v)




                '''
                ram_infos = env._get_ram()
                #print(ram_infos)
                
                #print(states)
                t = np.random.uniform() < 0.3
                unwrapped_env.restore_full_state(states[1])
                unwrapped_env.step(0)
                unwrapped_env.render()

                if t and len(state_list)>100:
                    print("restore")

                    idx = np.random.randint(len(state_list), size=1)[0]
                    print(idx)
                    print(len(state_list[idx][0]))
                    env.restore_full_state_by_idx(state_list[idx][1],1)
                    #actions = np.zeros_like(actions)
                    #env.step(actions)
                '''
                #print("ext_v:", values, "int_v:",int_v)
                #print(np.argmax(aprobs,axis=-1),aprobs)
                if args.alg=='arnd' and flag !=False:
                    
                    exp_b, dy_b, xr_states ,xrp_states = model.get_exp_bonus(last_obs, actions, obs, X_mean=obs_mean, X_var=obs_var,\
                       xr_S = xr_states, xrp_S= xrp_states, xr_M=dones, xrp_M=dones)


                    print(exp_b, dy_b)
                    


                    cam , aprobs_pred  = model.get_class_map(last_obs, obs, cam_action = actions[0])
                    scam = model.get_scam(last_obs)
                    #for i in range(18):
                    #    if i ==actions[0]:
                    #        continue
                    #    t, _, _  = model.get_class_map(last_obs, obs, cam_action = i )
                    #    cam +=t



                    print("pred_action_prob:", np.max(aprobs_pred) , "pred_action:", np.argmax(aprobs_pred))

                    classmap_vis = list(map(lambda x:(x-x.min()) / (x.max() - x.min()),cam))

                    #frame = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                    frame = cv2.resize(img, (84, 84), interpolation=cv2.INTER_AREA)
                    new_img = frame#frame[:, :, None]

                    frame = obs - last_obs
                    frame = frame[0,:,:,-1]
                    

                    plt.imshow(new_img)
                    plt.savefig('test.png')
                    plt.close()

                    plt.imshow(1-new_img)
                    #plt.imshow(frame, cmap=plt.cm.gray)
                    vis = classmap_vis[0]
                    #vis[vis < 0.4] =0
                    plt.imshow(vis, cmap=plt.cm.jet, alpha=0.5, interpolation='nearest', vmin=0, vmax=1)
                    #plt.show()
                    plt.savefig('cam_test.png')
                    plt.close()

                    
                    scam_vis = list(map(lambda x:(x-x.min()) / (x.max() - x.min()),scam))[0]


                    '''
                    scam_vis = np.expand_dims(scam_vis, axis=2)
                    scam_vis= np.tile(scam_vis,[1,1,3])

                    new_img = new_img.astype(float)
                    new_img /= new_img.max()
                    new_img = new_img + 3*scam_vis
                    new_img /= new_img.max()
                    
                    plt.imshow(new_img)
                    '''
                    #plt.imshow(frame, cmap=plt.cm.gray)
                    
                    plt.imshow(1-new_img)
                    plt.imshow(scam_vis, cmap=plt.cm.jet, alpha=0.5, interpolation='nearest', vmin=0, vmax=1)
                    #plt.show()
                    plt.savefig('scam_test.png')
                    plt.close()

                img = env.render()
                flag= True

                '''
                max_ac = np.argmax(aprobs, axis=-1)
                policy_cam = model.get_policy_cam(last_obs, cam_action = max_ac[0])
                print("max_ac:",max_ac)
                pcam_vis = list(map(lambda x:(x-x.min()) / (x.max() - x.min()),policy_cam))
                pcam_vis= pcam_vis[0]
                frame = cv2.resize(img, (84, 84), interpolation=cv2.INTER_AREA)
                new_img = frame


                pcam_vis = np.expand_dims(pcam_vis, axis=2)
                pcam_vis= np.tile(pcam_vis,[1,1,3])

                new_img = new_img.astype(float)
                new_img /= new_img.max()
                new_img = new_img + 3*pcam_vis
                new_img /= new_img.max()

                plt.imshow(new_img)
                #plt.imshow(frame, cmap=plt.cm.gray)
                #vis = pcam_vis[0]
                #vis[vis < 0.4] =0
                #plt.imshow(vis, cmap=plt.cm.jet, alpha=0.5, interpolation='nearest', vmin=0, vmax=1)
                #plt.show()
                plt.savefig('max_pcam_test.png')
                plt.close()
                '''


                '''
                next_scam  = model.get_scam(obs)
                frame = cv2.resize(img, (84, 84), interpolation=cv2.INTER_AREA)
                new_img = frame#frame[:, :, None]

                next_scam_vis = list(map(lambda x:(x-x.min()) / (x.max() - x.min()),next_scam))[0]

                #next_scam_vis[next_scam_vis <= 0.5] =0

                
                plt.imshow(1-new_img)

                plt.imshow(next_scam_vis, cmap=plt.cm.jet, alpha=0.5, interpolation='nearest', vmin=0, vmax=1)

                plt.savefig('next_scam_test.png')
                plt.close()
                '''
                    
                last_obs = obs


if __name__ == '__main__':
    main()

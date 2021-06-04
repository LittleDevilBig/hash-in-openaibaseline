import time
import functools
import tensorflow as tf
import numpy as np
from baselines import logger

from baselines.common import set_global_seeds, explained_variance
from baselines.common import tf_util
from baselines.common.policies import build_policy
from baselines.ppo2.ppo2 import safemean
from collections import deque
from baselines.a2c.utils import discount_with_dones
from baselines.common.runners import AbstractEnvRunner
from baselines.a2c.utils import conv, fc, conv_to_fc, batch_to_seq, seq_to_batch

from baselines.a2c.utils import Scheduler, find_trainable_variables
from baselines.a2c.runner import Runner

from tensorflow import losses


class Model(object):

    def __init__(self, policy, env, nsteps,
                 ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5, lr=7e-4,
                 alpha=0.99, epsilon=1e-5, total_timesteps=int(80e6), lrschedule='linear'):

        self.sess = tf_util.get_session()
        nenvs = env.num_envs
        nbatch = nenvs * nsteps

        with tf.variable_scope('a2c_model', reuse=tf.AUTO_REUSE):
            step_model = policy(nenvs, 1, self.sess)
            train_model = policy(nbatch, nsteps, self.sess)

        A = tf.placeholder(train_model.action.dtype, train_model.action.shape)
        ADV = tf.placeholder(tf.float32, [nbatch])
        R = tf.placeholder(tf.float32, [nbatch])
        LR = tf.placeholder(tf.float32, [], name="lr_")

        neglogpac = train_model.pd.neglogp(A)
        entropy = tf.reduce_mean(train_model.pd.entropy())

        pg_loss = tf.reduce_mean(ADV * neglogpac)
        vf_loss = losses.mean_squared_error(tf.squeeze(train_model.vf), R)

        loss = pg_loss - entropy * ent_coef + vf_loss * vf_coef

        params = find_trainable_variables("a2c_model")
        grads = tf.gradients(loss, params)
        if max_grad_norm is not None:
            grads, grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        grads = list(zip(grads, params))
        trainer = tf.train.RMSPropOptimizer(learning_rate=LR, decay=alpha, epsilon=epsilon)
        _train = trainer.apply_gradients(grads)

        lr = Scheduler(v=lr, nvalues=total_timesteps, schedule=lrschedule)

        with tf.variable_scope('ae_model', reuse=tf.AUTO_REUSE):
            self.inputs_ = tf.placeholder(tf.float32, (None, 84, 84, 4), name='inputs')
            self.scaled_images = tf.cast(self.inputs_, tf.float32) / 255.
            activ = tf.tanh
            print(self.scaled_images.shape)
            h = activ(conv(self.scaled_images, 'd1', nf=32, rf=8, stride=4, init_scale=np.sqrt(2)))
            print('0:', h.shape)
            h2 = activ(conv(h, 'd2', nf=64, rf=4, stride=2, init_scale=np.sqrt(2)))
            print('1:', h2.shape)
            h3 = activ(conv(h2, 'd3', nf=16, rf=5, stride=1, init_scale=np.sqrt(2)))
            print('2:', h3.shape)
            self.h3 = conv_to_fc(h3)
            print('3:', self.h3.shape)
            h4 = tf.reshape(self.h3, (-1, 5, 5, 16))
            print('40:', h4.shape)
            upsample1 = tf.image.resize_images(h4, size=(9, 9), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            conv4 = tf.layers.conv2d(inputs=upsample1, filters=64, kernel_size=(3, 3), padding='same',
                                     activation=tf.nn.relu)
            print('5:', conv4.shape)
            upsample2 = tf.image.resize_images(conv4, size=(20, 20), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            conv5 = tf.layers.conv2d(inputs=upsample2, filters=32, kernel_size=(3, 3), padding='same',
                                     activation=tf.nn.relu)
            print('6:', conv5.shape)
            upsample3 = tf.image.resize_images(conv5, size=(84, 84), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            conv6 = tf.layers.conv2d(inputs=upsample3, filters=4, kernel_size=(3, 3), padding='same',
                                     activation=tf.nn.relu)
            print('7:', conv6.shape)
            logits = tf.layers.conv2d(inputs=conv6, filters=4, kernel_size=(3, 3), padding='same', activation=None)
            print('8:', logits.shape)
        # Pass logits through sigmoid to get reconstructed image
        # decoded = tf.nn.sigmoid(logits)
        # Pass logits through sigmoid and calculate the cross-entropy loss
        self.loss = tf.reduce_mean(tf.square(self.scaled_images - logits))+1.5*(tf.reduce_mean(tf.square(self.h3-tf.math.sign(self.h3))))
        # Get cost and define the optimizer
        params1 = find_trainable_variables("ae_model")
        grads1 = tf.gradients(self.loss, params1)
        if max_grad_norm is not None:
            grads1, grad_norm1 = tf.clip_by_global_norm(grads1, max_grad_norm)
        grads1 = list(zip(grads1, params1))
        self.opt = tf.train.RMSPropOptimizer(learning_rate=LR, decay=alpha, epsilon=epsilon)
        self._train = self.opt.apply_gradients(grads1)
        self.count = {}
        self.A = np.random.normal(size=(256, 400))

        def train(obs, states, rewards, masks, actions, values):
            advs = rewards - values
            for step in range(len(obs)):
                cur_lr = lr.value()

            td_map = {train_model.X: obs, A: actions, ADV: advs, R: rewards, LR: cur_lr}
            if states is not None:
                td_map[train_model.S] = states
                td_map[train_model.M] = masks
            policy_loss, value_loss, policy_entropy, _ = self.sess.run(
                [pg_loss, vf_loss, entropy, _train],
                td_map
            )
            return policy_loss, value_loss, policy_entropy

        self.train = train
        self.train_model = train_model
        self.step_model = step_model
        self.step = step_model.step
        self.value = step_model.value
        self.initial_state = step_model.initial_state
        self.save = functools.partial(tf_util.save_variables, sess=self.sess)
        self.load = functools.partial(tf_util.load_variables, sess=self.sess)
        tf.global_variables_initializer().run(session=self.sess)

    def train_ae(self, input_train,lr,epochs):
        print('!!!!!!!!!!!!')
        file2 = open('aelossoutput.txt', 'a')
        batch_size = 1024

        for e in range(epochs):
            for ii in range(input_train.shape[0] // batch_size):
                batch = epoch_input = input_train[ii * batch_size: (ii + 1) * batch_size]
                # Get images from the batch
                imgs = batch.reshape((-1, 84, 84, 4))
                print(type(imgs))

                batch_cost, _ = self.sess.run([self.loss, self._train], feed_dict={self.inputs_: imgs, 'lr_:0': lr})
                print("Epoch: {}/{}...".format(e + 1, epochs),
                      "Training loss: {:.4f}".format(batch_cost))
                file2.write(str(batch_cost) + '\n')

    def getEncodedImage(self, state):
        # self._session.run(self._training)
        state = np.reshape(state, (-1, 84, 84, 4))
        h3 = self.sess.run(self.h3, feed_dict={self.inputs_: state})
        # print(h3.shape)
        return h3

    def counting(self, state):
        g = self.getEncodedImage(state)

        signResult = np.sign(g)

        for i in range(signResult.shape[0]):
            Index = tuple(signResult[i])
            if Index in self.count:
                self.count[Index] += 1
            else:
                self.count[Index] = 1
        current_count = []
        for i in range(signResult.shape[0]):
            Index = tuple(signResult[i])
            current_count.append(self.count[Index])
        return np.asarray(current_count)


def learn(
        network,
        env,
        seed=None,
        nsteps=5,
        total_timesteps=int(80e6),
        vf_coef=0.5,
        ent_coef=0.01,
        max_grad_norm=0.5,
        lr=7e-4,
        lrschedule='linear',
        epsilon=1e-5,
        alpha=0.99,
        gamma=0.99,
        log_interval=100,
        load_path=None,
        save_path=None,
        **network_kwargs):
    '''
    Main entrypoint for A2C algorithm. Train a policy with given network architecture on a given environment using a2c algorithm.

    Parameters:
    -----------

    network:            policy network architecture. Either string (mlp, lstm, lnlstm, cnn_lstm, cnn, cnn_small, conv_only - see baselines.common/models.py for full list)
                        specifying the standard network architecture, or a function that takes tensorflow tensor as input and returns
                        tuple (output_tensor, extra_feed) where output tensor is the last network layer output, extra_feed is None for feed-forward
                        neural nets, and extra_feed is a dictionary describing how to feed state into the network for recurrent neural nets.
                        See baselines.common/policies.py/lstm for more details on using recurrent nets in policies


    env:                RL environment. Should implement interface similar to VecEnv (baselines.common/vec_env) or be wrapped with DummyVecEnv (baselines.common/vec_env/dummy_vec_env.py)


    seed:               seed to make random number sequence in the alorightm reproducible. By default is None which means seed from system noise generator (not reproducible)

    nsteps:             int, number of steps of the vectorized environment per update (i.e. batch size is nsteps * nenv where
                        nenv is number of environment copies simulated in parallel)

    total_timesteps:    int, total number of timesteps to train on (default: 80M)

    vf_coef:            float, coefficient in front of value function loss in the total loss function (default: 0.5)

    ent_coef:           float, coeffictiant in front of the policy entropy in the total loss function (default: 0.01)

    max_gradient_norm:  float, gradient is clipped to have global L2 norm no more than this value (default: 0.5)

    lr:                 float, learning rate for RMSProp (current implementation has RMSProp hardcoded in) (default: 7e-4)

    lrschedule:         schedule of learning rate. Can be 'linear', 'constant', or a function [0..1] -> [0..1] that takes fraction of the training progress as input and
                        returns fraction of the learning rate (specified as lr) as output

    epsilon:            float, RMSProp epsilon (stabilizes square root computation in denominator of RMSProp update) (default: 1e-5)

    alpha:              float, RMSProp decay parameter (default: 0.99)

    gamma:              float, reward discounting parameter (default: 0.99)

    log_interval:       int, specifies how frequently the logs are printed out (default: 100)

    **network_kwargs:   keyword arguments to the policy / network builder. See baselines.common/policies.py/build_policy and arguments to a particular type of network
                        For instance, 'mlp' network architecture has arguments num_hidden and num_layers.

    '''

    save_interval = 1e6

    if save_path is not None:
        logger.log("****************save path: {}************************".format(save_path))
    set_global_seeds(seed)

    lr = 1e-4
    nenvs = env.num_envs
    policy = build_policy(env, network, **network_kwargs)

    model = Model(policy=policy, env=env, nsteps=nsteps, ent_coef=ent_coef, vf_coef=vf_coef,
                  max_grad_norm=max_grad_norm, lr=lr, alpha=alpha, epsilon=epsilon, total_timesteps=total_timesteps,
                  lrschedule=lrschedule)
    if load_path is not None:
        model.load(load_path)
    runner = Runner(env, model, nsteps=nsteps, gamma=gamma)
    epinfobuf = deque(maxlen=100)

    nbatch = nenvs * nsteps

    file = open('AEHero.txt', 'a')
    '''
    obss = []
    for i in range(500):
        obs, states, rewards, masks, actions, values, epinfos = runner.run(model)
        obss.append(obs)
    x_train = np.asarray(obss, dtype=np.float32)
    # print(x_train.shape)
    x_train = np.reshape(x_train, (-1, 84, 84, 4))
    model.train_ae(input_train=x_train,lr=0.01)'''

    save_interval = int(save_interval / nbatch)

    tstart = time.time()
    obs2=[]
    for update in range(1, total_timesteps // nbatch + 1):
        obs, states, rewards, masks, actions, values, epinfos = runner.run(model)
        epinfobuf.extend(epinfos)
        obs2.append(obs)

        if update%2000==0:
            x2_train = np.asarray(obs2, dtype=np.float32)
            x2_train = np.reshape(x2_train, (-1, 84, 84, 4))
            model.train_ae(input_train=x2_train,lr=5e-7,epochs=5)
            obs2=[]

        policy_loss, value_loss, policy_entropy = model.train(obs, states, rewards, masks, actions, values)
        nseconds = time.time() - tstart
        fps = int((update * nbatch) / nseconds)
        if update % log_interval == 0 or update == 1:
            ev = explained_variance(values, rewards)
            logger.record_tabular("nupdates", update)
            logger.record_tabular("total_timesteps", update * nbatch)
            logger.record_tabular("fps", fps)
            logger.record_tabular("policy_entropy", float(policy_entropy))
            logger.record_tabular("value_loss", float(value_loss))
            logger.record_tabular("explained_variance", float(ev))
            logger.record_tabular("eprewmean", safemean([epinfo['r'] for epinfo in epinfobuf]))
            file.write(str(safemean([epinfo['r'] for epinfo in epinfobuf])) + '\n')
            logger.dump_tabular()
        if save_path is not None and (update%save_interval==0 or update ==1):
            logger.log("save model")
            model.save('{}_{}'.format(save_path,str(update*nbatch)))
    env.close()
    return model


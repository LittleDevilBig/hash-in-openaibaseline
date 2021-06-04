import numpy as np
from baselines.a2c.utils import discount_with_dones
from baselines.common.runners import AbstractEnvRunner
from baselines.a2c.utils import conv, fc, conv_to_fc, batch_to_seq, seq_to_batch
import tensorflow as tf

'''
class Autoencoder(object):

    def __init__(self):
        learning_rate = 0.001

        """
        CNN from Nature paper.
        """
        self.inputs_ = tf.placeholder(tf.float32, (None, 84, 84, 4), name='inputs')
        self.scaled_images = tf.cast(self.inputs_, tf.float32) / 255.
        activ = tf.nn.relu
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
        print('4:', h4.shape)
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
        decoded = tf.nn.sigmoid(logits)
        # Pass logits through sigmoid and calculate the cross-entropy loss
        self.loss = tf.reduce_mean(tf.square(self.scaled_images - logits))
        # Get cost and define the optimizer
        self.opt = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)
        self.count = {}
        self.A = np.random.normal(size=(256, 400))

        self._training = tf.initialize_all_variables()
        self._session = tf.Session()

    def train(self, input_train):
        file2 = open('aelossoutput.txt', 'a')
        self._session.run(self._training)
        epochs = 100
        batch_size = 256


        for e in range(epochs):
            for ii in range(input_train.shape[0] // batch_size):
                batch = epoch_input = input_train[ii * batch_size: (ii + 1) * batch_size]
                # Get images from the batch
                imgs = batch.reshape((-1, 84, 84, 4))

                batch_cost, _ = self._session.run([self.loss, self.opt], feed_dict={self.inputs_: imgs})
                print("Epoch: {}/{}...".format(e + 1, epochs),
                      "Training loss: {:.4f}".format(batch_cost))
                file2.write(str(batch_cost) + '\n')

    def getEncodedImage(self, state):
        #self._session.run(self._training)
        state = np.reshape(state, (-1, 84, 84, 4))
        h3 = self._session.run(self.h3, feed_dict={self.inputs_: state})
        #print(h3.shape)
        return h3

    def counting(self, state):
        g = self.getEncodedImage(state)
        #print('g', g.shape)

        #print(A.shape)
        x = np.dot(g, self.A.T)
        #print(x.shape)
        signResult = np.sign(x)

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
'''


class Runner(AbstractEnvRunner):

    def __init__(self, env, model, nsteps=5, gamma=0.99):

        # reset env , and let obs := init_state
        super().__init__(env=env, model=model, nsteps=nsteps)
        self.gamma = gamma
        self.batch_action_shape = [x if x is not None else -1 for x in model.train_model.action.shape.as_list()]
        self.ob_dtype = model.train_model.X.dtype.as_numpy_dtype

    '''
    def run_ae(self,model):
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones = [], [], [], [], []
        epinfos = []

        for k in range(500):
            for n in range(self.nsteps):
                actions, values, states, _ = self.model.step(self.obs, S=self.states, M=self.dones)
                mb_obs.append(np.copy(self.obs))
                mb_actions.append(actions)
                mb_values.append(values)
                mb_dones.append(self.dones)
                obs, rewards, dones, infos = self.env.step(actions)
                for info in infos:
                    maybeepinfo = info.get('episode')
                    if maybeepinfo: epinfos.append(maybeepinfo)
                self.states = states
                self.dones = dones
                for i, done in enumerate(dones):
                    if done:
                        self.obs[i] = self.obs[i] * 0
                    if 'episode' in infos[i] and i == 0:
                        print("env_id:{} episode_rewards:{}\n".format(i, infos[i]['episode']['r']))
                mb_obs.append(self.obs)
                self.obs = obs

                mb_rewards.append(rewards)

        x_train = np.asarray(mb_obs, dtype=np.float32)
        #print(x_train.shape)
        x_train = np.reshape(x_train, (-1, 84, 84, 4))
        model.train_ae(input_train=x_train)
    '''

    def run(self, model):
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones = [], [], [], [], []
        epinfos = []
        beta = 0.01

        mb_states = self.states
        for n in range(self.nsteps):
            actions, values, states, _ = self.model.step(self.obs, S=self.states, M=self.dones)
            mb_obs.append(np.copy(self.obs))
            mb_actions.append(actions)
            mb_values.append(values)
            mb_dones.append(self.dones)
            obs, rewards, dones, infos = self.env.step(actions)
            for info in infos:
                maybeepinfo = info.get('episode')
                if maybeepinfo: epinfos.append(maybeepinfo)
            self.states = states
            self.dones = dones
            for i, done in enumerate(dones):
                if done:
                    self.obs[i] = self.obs[i] * 0
                if 'episode' in infos[i] and i == 0:
                    print("env_id:{} episode_rewards:{}\n".format(i, infos[i]['episode']['r']))

            self.obs = obs

            mb_rewards.append(rewards)

        mb_dones.append(self.dones)
        # batch of steps to batch of rollouts

        mb_obs = np.asarray(mb_obs, dtype=self.ob_dtype).swapaxes(1, 0).reshape(self.batch_ob_shape)
        # print(mb_obs.shape)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32).swapaxes(1, 0)
        # print(mb_rewards.shape)
        count = model.counting(mb_obs)
        count = np.reshape(count, [-1, self.nsteps])
        hash_rewards = beta / (count ** 0.5)
        mb_rewards += hash_rewards
        mb_actions = np.asarray(mb_actions, dtype=self.model.train_model.action.dtype.name).swapaxes(1, 0)
        mb_values = np.asarray(mb_values, dtype=np.float32).swapaxes(1, 0)
        mb_dones = np.asarray(mb_dones, dtype=np.bool).swapaxes(1, 0)
        mb_masks = mb_dones[:, :-1]
        mb_dones = mb_dones[:, 1:]

        # print("dones_shape:",mb_dones.shape)

        if self.gamma > 0.0:
            # discount/bootstrap off value fn
            last_values = self.model.value(self.obs, S=self.states, M=self.dones).tolist()
            for n, (rewards, dones, value) in enumerate(zip(mb_rewards, mb_dones, last_values)):
                rewards = rewards.tolist()
                dones = dones.tolist()
                if dones[-1] == 0:
                    rewards = discount_with_dones(rewards + [value], dones + [0], self.gamma)[:-1]
                else:
                    rewards = discount_with_dones(rewards, dones, self.gamma)

                mb_rewards[n] = rewards

        mb_actions = mb_actions.reshape(self.batch_action_shape)

        mb_rewards = mb_rewards.flatten()
        mb_values = mb_values.flatten()
        mb_masks = mb_masks.flatten()
        return mb_obs, mb_states, mb_rewards, mb_masks, mb_actions, mb_values, epinfos

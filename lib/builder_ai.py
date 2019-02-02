# Using PPO algorithm from spinningup.openai.com
# (Wiring a brain from Elon Musk to the bot)


import os
import os.path
import time

import numpy as np
import tensorflow as tf

import gym
import spinup.algos.ppo.core as core
import spinup.algos.ppo.ppo as ppo
from spinup.utils.logx import EpochLogger
from spinup.utils.mpi_tf import MpiAdamOptimizer, sync_all_params
from spinup.utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs


class Builder(object):
    def __init__(self, args = {}):
        self.bot = None
        if "bot" in args:
            bot = args["bot"]

        self.epoch = 0
        self.step = 0

        self.actor_critic=core.mlp_actor_critic
        self.ac_kwargs=dict(hidden_sizes=[64]*2)
        self.seed=0 
        self.steps_per_epoch=10000        
        self.epochs=5
        self.gamma=0.99
        self.clip_ratio=0.2
        self.pi_lr=3e-4
        self.vf_lr=1e-3
        self.train_pi_iters=80
        self.train_v_iters=80
        self.lam=0.97
        self.max_ep_len=1000
        self.target_kl=0.01
        self.logger_kwargs={}
        self.save_freq=1

        map_name = "unknown"
        if bot is not None:
            map_name = bot.map_name
        self.logger_kwargs = {"output_dir":f".\\{map_name}\\ai_data","exp_name":"builder_ai"}
        
        self.logger = EpochLogger(**self.logger_kwargs)
        
        #self.logger.save_config(locals())
        self.logger.save_config(self.__dict__)

        seed = self.seed
        seed += 10000 * proc_id()
        tf.set_random_seed(seed)
        np.random.seed(seed)

        #env = env_fn()
        self.env = BuilderEnv(args = {"bot":self.bot})
        obs_dim = self.env.observation_space.shape
        act_dim = self.env.action_space.shape
    
        # Share information about action space with policy architecture
        self.ac_kwargs['action_space'] = self.env.action_space

        print(str(self.env.observation_space))
        print(str(self.env.action_space))

        print(str(type(self.env.observation_space)))
        print(str(type(self.env.action_space)))


        # Inputs to computation graph
        self.x_ph, self.a_ph = core.placeholders_from_spaces(self.env.observation_space, self.env.action_space)
        self.adv_ph, self.ret_ph, self.logp_old_ph = core.placeholders(None, None, None)

        # Main outputs from computation graph
        self.pi, self.logp, self.logp_pi, self.v = self.actor_critic(self.x_ph, self.a_ph, **self.ac_kwargs)

        # Need all placeholders in *this* order later (to zip with data from buffer)
        self.all_phs = [self.x_ph, self.a_ph, self.adv_ph, self.ret_ph, self.logp_old_ph]

        # Every step, get: action, value, and logprob
        self.get_action_ops = [self.pi, self.v, self.logp_pi]

        # Experience buffer
        self.local_steps_per_epoch = int(self.steps_per_epoch / num_procs()) 
        self.buf = ppo.PPOBuffer(obs_dim, act_dim, self.local_steps_per_epoch, self.gamma, self.lam) # *2 is to create a lot of extra space in the buffer, hopefully?

        # Count variables
        var_counts = tuple(core.count_vars(scope) for scope in ['pi', 'v'])
        self.logger.log('\nNumber of parameters: \t pi: %d, \t v: %d\n'%var_counts)

        # PPO objectives
        self.ratio = tf.exp(self.logp - self.logp_old_ph)          # pi(a|s) / pi_old(a|s)
        self.min_adv = tf.where(self.adv_ph>0, (1+self.clip_ratio)*self.adv_ph, (1-self.clip_ratio)*self.adv_ph)
        self.pi_loss = -tf.reduce_mean(tf.minimum(self.ratio * self.adv_ph, self.min_adv))
        self.v_loss = tf.reduce_mean((self.ret_ph - self.v)**2)

        # Info (useful to watch during learning)
        self.approx_kl = tf.reduce_mean(self.logp_old_ph - self.logp)      # a sample estimate for KL-divergence, easy to compute
        self.approx_ent = tf.reduce_mean(-self.logp)                  # a sample estimate for entropy, also easy to compute
        self.clipped = tf.logical_or(self.ratio > (1+self.clip_ratio), self.ratio < (1-self.clip_ratio))
        self.clipfrac = tf.reduce_mean(tf.cast(self.clipped, tf.float32))

        print(f"pi_lr:{self.pi_lr}, pi_loss:{self.pi_loss}")

        # Optimizers
        self.train_pi = MpiAdamOptimizer(learning_rate=self.pi_lr).minimize(self.pi_loss)
        self.train_v = MpiAdamOptimizer(learning_rate=self.vf_lr).minimize(self.v_loss)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        # Sync params across processes
        self.sess.run(sync_all_params())

        # Setup model saving
        self.logger.setup_tf_saver(self.sess, inputs={'x': self.x_ph}, outputs={'pi': self.pi, 'v': self.v})

        self.start_time = time.time()
        self.o, self.r, self.d, self.ep_ret, self.ep_len = self.env.reset(args={}), 0, False, 0, 0
        
        print(f"o:{self.o}, type:{type(self.o)}")

        self.epoch = 0
        self.t = 0

        self.load()

    def reset(self, o):
        self.env.reset(args = {"o":o})

    # since the env is calling us for action instead of the other way around we're making a few changes here
    # specifically, beraking up the funciton into a get_action and a wrap_up_action set
    def get_action(self, new_o = None):
        if new_o is not None:
            self.o = new_o
        # Main loop: collect experience in env and update/log each epoch
        self.a, self.v_t, self.logp_t = self.sess.run(self.get_action_ops, feed_dict={self.x_ph: self.o.reshape(1,-1)})

        # save and log
        self.buf.store(self.o, self.a, self.r, self.v_t, self.logp_t)
        self.logger.store(VVals=self.v_t)

        return self.a[0]

    def wrap_up_action(self, o, r, d, _):
        logger = self.logger

        if self.epoch < self.epochs:
            if self.t < self.local_steps_per_epoch:

                # o = state space
                # r = reward from the step
                # d = done or not
                # _ = {} of something random not used
                self.o, self.r, self.d, self._ = o, r, d, _

                self.ep_ret += r
                self.ep_len += 1

                terminal = d or (self.ep_len == self.max_ep_len)
                if terminal or (self.t==self.local_steps_per_epoch-1):
                    print(f"here we are at the end of an episode or epoch: t={self.t} of {self.local_steps_per_epoch-1}")
                    if not(terminal):
                        print('Warning: trajectory cut off by epoch at %d steps.'%self.ep_len)
                        
                        # Save model
                        if (self.epoch % self.save_freq == 0) or (self.epoch == self.epochs-1):
                            self.save()

                        # Perform PPO update!
                        self.update()

                        # Log info about epoch
                        logger.log_tabular('Epoch', self.epoch)
                        logger.log_tabular('EpRet', with_min_and_max=True)
                        logger.log_tabular('EpLen', average_only=True)
                        logger.log_tabular('VVals', with_min_and_max=True)
                        logger.log_tabular('TotalEnvInteracts', (self.epoch+1)*self.steps_per_epoch)
                        logger.log_tabular('LossPi', average_only=True)
                        logger.log_tabular('LossV', average_only=True)
                        logger.log_tabular('DeltaLossPi', average_only=True)
                        logger.log_tabular('DeltaLossV', average_only=True)
                        logger.log_tabular('Entropy', average_only=True)
                        logger.log_tabular('KL', average_only=True)
                        logger.log_tabular('ClipFrac', average_only=True)
                        logger.log_tabular('StopIter', average_only=True)
                        logger.log_tabular('Time', time.time()-self.start_time)
                        logger.dump_tabular()
                
                        self.epoch += 1
                        self.t = 0
                        # this epoch has completed, resign the game and let's move on to the next one
                        return 1

                    # if trajectory didn't reach terminal state, bootstrap value target
                    last_val = r if d else self.sess.run(self.v, feed_dict={self.x_ph: o.reshape(1,-1)})
                    self.buf.finish_path(last_val)
                    if terminal:
                        # only save EpRet / EpLen if trajectory finished
                        logger.store(EpRet=self.ep_ret, EpLen=self.ep_len)
                    self.o, self.r, self.d, self.ep_ret, self.ep_len = self.env.reset(args={}), 0, False, 0, 0

                self.t += 1

        else:
            # epochs have completed, end simulation
            return -1

        return 0

    def save(self, args = {}):
        logger = self.logger
        env = self.env
        # TODO: How does this work, if it does work?
        logger.save_state({'env': env}, None)
        data_name = self.logger_kwargs["output_dir"]+"\\training_model"
        saver = tf.train.Saver()
        saver.save(self.sess, data_name)
        print("saved the model to: " + data_name)
        return

    # TODO: When we load the model, which learning parameters do we need to set for our distance down the rabbit hole?
    def load(self, args = {}):
        data_name = self.logger_kwargs["output_dir"]+"\\training_model"
        if os.path.exists(data_name+".index"):
            saver = tf.train.Saver()
            saver.restore(self.sess, data_name)
            print("loaded up some data, not sure how to prove it")
        else:
            print("no data to load, ai is starting from stupid")
        return

    def update(self):
        logger = self.logger

        inputs = {k:v for k,v in zip(self.all_phs, self.buf.get())}
        pi_l_old, v_l_old, ent = self.sess.run([self.pi_loss, self.v_loss, self.approx_ent], feed_dict=inputs)

        # Training
        for i in range(self.train_pi_iters):
            _, kl = self.sess.run([self.train_pi, self.approx_kl], feed_dict=inputs)
            kl = mpi_avg(kl)
            if kl > 1.5 * self.target_kl:
                logger.log('Early stopping at step %d due to reaching max kl.'%i)
                break
        logger.store(StopIter=i)
        for _ in range(self.train_v_iters):
            self.sess.run(self.train_v, feed_dict=inputs)

        # Log changes from update
        pi_l_new, v_l_new, kl, cf = self.sess.run([self.pi_loss, self.v_loss, self.approx_kl, self.clipfrac], feed_dict=inputs)
        logger.store(LossPi=pi_l_old, LossV=v_l_old, 
                     KL=kl, Entropy=ent, ClipFrac=cf,
                     DeltaLossPi=(pi_l_new - pi_l_old),
                     DeltaLossV=(v_l_new - v_l_old))


class BuilderEnv():
    def __init__(self, args = {}):
        self.bot = None
        self.o = np.zeros([8],dtype=int)
        if "bot" in args:
            bot = args["bot"]

        self.reset()
        return

    def reset(self, args = {}):
        o = None
        
        if "o" in args:
            o = args["o"]
            self.o = o
            

        # get this stuff from the bot
        self.observation_space = gym.spaces.Box(low=0, high=100000, shape=(8,), dtype=np.int)
        self.action_space = gym.spaces.Discrete(8)
        if o is not None:
            return o
        elif self.o is not None:
            return self.o
        else:
            return


    def step(self):
        print("You shouldn't be here")
        return

    
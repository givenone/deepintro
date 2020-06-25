import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

import gym
import logz
import os
import os.path as osp
import time
import inspect
from multiprocessing import Process

#============================================================================================#
# Utilities
#============================================================================================#

#========================================================================================#
#  ----------Implement vanilla policy gradien algorithm for CartPole environment (discrete action environment)---------
#========================================================================================# 

#========================================================================================#
#                           ----------PROBLEM 1. Build network----------
#========================================================================================#  
def build_mlp(input_placeholder, output_size, scope, n_layers, size, activation=tf.tanh, output_activation=None):
    """
        Builds a feedforward neural network
        
        arguments:
            input_placeholder: placeholder variable for the state (batch_size, input_size)
            output_size: size of the output layer
            scope: variable scope of the network
            n_layers: number of hidden layers
            size: dimension of the hidden layer
            activation: activation of the hidden layers
            output_activation: activation of the ouput layers

        returns:
            output placeholder of the network (the result of a forward pass) 

        Hint: use tf.layers.dense    
    """
    # YOUR CODE HERE

    with tf.variable_scope(scope) :
        x = input_placeholder
        for i in range(n_layers) :
            x = tf.layers.dense(inputs=x, units=size, activation=activation)
        output_placeholder = tf.layers.dense(inputs=x, units=output_size, activation=output_activation)
    #raise NotImplementedError
    return output_placeholder

def pathlength(path):
    return len(path["reward"])

def setup_logger(logdir, locals_):
    # Configure output directory for logging
    logz.configure_output_dir(logdir)
    # Log experimental parameters
    args = inspect.getargspec(train_PG)[0]
    params = {k: locals_[k] if k in locals_ else None for k in args}
    logz.save_params(params)

#============================================================================================#
# Policy Gradient
#============================================================================================#

class Agent(object):
    def __init__(self, computation_graph_args, sample_trajectory_args, estimate_return_args, agent_args):
        super(Agent, self).__init__()
        self.ob_dim = computation_graph_args['ob_dim']
        self.ac_dim = computation_graph_args['ac_dim']
        self.size = computation_graph_args['size']
        self.n_layers = computation_graph_args['n_layers']
        self.learning_rate = computation_graph_args['learning_rate']

        self.animate = sample_trajectory_args['animate']
        self.max_path_length = sample_trajectory_args['max_path_length']
        self.min_timesteps_per_batch = sample_trajectory_args['min_timesteps_per_batch']

        self.gamma = estimate_return_args['gamma']

        self.nn_baseline = agent_args['nn_baseline'] 
        self.reward_to_go = agent_args['reward_to_go'] 


    def init_tf_sess(self):
        tf_config = tf.ConfigProto(inter_op_parallelism_threads=1, intra_op_parallelism_threads=1) 
        self.sess = tf.Session(config=tf_config)
        self.sess.__enter__() # equivalent to `with self.sess:`
        tf.global_variables_initializer().run() #pylint: disable=E1101

    #========================================================================================#
    #                           ----------PROBLEM 2. Define placeholders----------
    #========================================================================================#
    def define_placeholders(self):
        """
            Placeholders for batch observations / actions / q values in policy gradient 
            loss function.
            See Agent.build_computation_graph for notation

            returns:
                sy_ob_no: placeholder for observations
                sy_ac_na: placeholder for actions
                sy_q_n: placeholder for estimated q values
        """

        sy_ob_no = tf.placeholder(shape=[None, self.ob_dim], name="ob", dtype=tf.float32)
        sy_ac_na = tf.placeholder(shape=[None], name="ac", dtype=tf.int32) 
        sy_targets_n = tf.placeholder(shape=[None], name="baseline_target", dtype=tf.float32)
        # YOUR CODE HERE
        sy_q_n = tf.placeholder(shape=[None], name="q_values", dtype=tf.float32)
        return sy_ob_no, sy_ac_na, sy_q_n, sy_targets_n

    #========================================================================================#
    #                           ----------PROBLEM 3. Define Forward Pass----------
    #========================================================================================#
    def policy_forward_pass(self, sy_ob_no):
        """ Constructs the symbolic operation for the policy network outputs,
            which are the parameters of the policy distribution p(a|s)

            arguments:
                sy_ob_no: (batch_size, self.ob_dim)

            returns:
                the parameters of the policy.

                if discrete, the parameters are the logits of a categorical distribution
                    over the actions
                    sy_logits_na: (batch_size, self.ac_dim)

            Hint: use the 'build_mlp' function to output the logits (in the discrete case).
                Pass in self.n_layers for the 'n_layers' argument, and
                pass in self.size for the 'size' argument.
        """
        # YOUR CODE HERE
        sy_logits_na = build_mlp(input_placeholder=sy_ob_no, output_size=self.ac_dim, scope="policy", n_layers=self.n_layers, size=self.size)
        return sy_logits_na

    def baseline_forward_pass(self, sy_ob_no):
        return tf.squeeze(build_mlp(sy_ob_no, output_size=1, scope='nn_baseline', n_layers=self.n_layers, size=self.size))

    #========================================================================================#
    #                           ----------PROBLEM 4. Sample action----------
    #========================================================================================#
    def sample_action(self, policy_parameters):
        """ Constructs a symbolic operation for stochastically sampling from the policy
            distribution

            arguments:
                policy_parameters
                    if discrete: logits of a categorical distribution over actions 
                        sy_logits_na: (batch_size, self.ac_dim)

            returns:
                sy_sampled_ac: 
                    if discrete: (batch_size,)

        """
        sy_logits_na = policy_parameters
        # YOUR CODE HERE
        sy_sampled_ac = tf.squeeze(tf.multinomial(sy_logits_na, 1), axis=[1])
        return sy_sampled_ac

    #========================================================================================#
    #                           ----------PROBLEM 5. Compute log probability----------
    #========================================================================================#
    def get_log_prob(self, policy_parameters, sy_ac_na):
        """ Constructs a symbolic operation for computing the log probability of a set of actions
            that were actually taken according to the policy

            arguments:
                policy_parameters
                    if discrete: logits of a categorical distribution over actions 
                        sy_logits_na: (batch_size, self.ac_dim)

                sy_ac_na: 
                    if discrete: (batch_size,)

            returns:
                sy_logprob_n: (batch_size)

            Hint:
                For the discrete case, use the log probability under a categorical distribution.
        """
        sy_logits_na = policy_parameters
        # YOUR CODE HERE
        sy_logprob_n = -tf.nn.sparse_softmax_cross_entropy_with_logits(labels=sy_ac_na, logits=sy_logits_na)
        return sy_logprob_n

    def build_computation_graph(self):
        """
            Notes on notation:
            
            Symbolic variables have the prefix sy_, to distinguish them from the numerical values
            that are computed later in the function
            
            Prefixes and suffixes:
            ob - observation 
            ac - action
            _no - this tensor should have shape (batch self.size /n/, observation dim)
            _na - this tensor should have shape (batch self.size /n/, action dim)
            _n  - this tensor should have shape (batch self.size /n/)
            
            Note: batch self.size /n/ is defined at runtime, and until then, the shape for that axis
            is None

            ----------------------------------------------------------------------------------
            loss: a function of self.sy_logprob_n and self.sy_q_n that we will differentiate
                to get the policy gradient.
        """
        self.sy_ob_no, self.sy_ac_na, self.sy_q_n, self.sy_targets_n = self.define_placeholders()

        # The policy takes in an observation and produces a distribution over the action space
        self.policy_parameters = self.policy_forward_pass(self.sy_ob_no)

        # We can sample actions from this action distribution.
        # This will be called in Agent.sample_trajectory() where we generate a rollout.
        self.sy_sampled_ac = self.sample_action(self.policy_parameters)

        # We can also compute the logprob of the actions that were actually taken by the policy
        # This is used in the loss function.
        self.sy_logprob_n = self.get_log_prob(self.policy_parameters, self.sy_ac_na)

        #========================================================================================#
        #                           ----------PROBLEM 6. Compute loss function----------
        # Loss Function and Training Operation
        #========================================================================================#
        # YOUR CODE HERE
        
        loss = -tf.reduce_mean(tf.multiply(self.sy_logprob_n, self.sy_q_n))
        self.update_op = tf.train.AdamOptimizer(self.learning_rate).minimize(loss)

        if self.nn_baseline:
            self.sy_bl_pred = self.baseline_forward_pass(self.sy_ob_no)
            # TODO: define the loss that should be optimized for training the baseline
            # HINT1: use tf.losses.mean_squared_error
            # HINT2: we want predictions (self.sy_bl_pred) to be as close as possible to the labels (self.sy_targets_n)
            
            baseline_loss = tf.losses.mean_squared_error(self.sy_bl_pred, self.sy_targets_n)

            # TODO: define what exactly the optimizer should minimize when updating the baseline
            self.baseline_update_op = tf.train.AdamOptimizer(self.learning_rate).minimize(baseline_loss)

    def sample_trajectories(self, itr, env):
        # Collect paths until we have enough timesteps
        timesteps_this_batch = 0
        paths = []
        while True:
            animate_this_episode=(len(paths)==0 and (itr % 10 == 0) and self.animate)
            path = self.sample_trajectory(env, animate_this_episode)
            paths.append(path)
            timesteps_this_batch += pathlength(path)
            if timesteps_this_batch > self.min_timesteps_per_batch:
                break
        return paths, timesteps_this_batch

    def sample_trajectory(self, env, animate_this_episode):
        ob = env.reset()
        obs, acs, rewards = [], [], []
        steps = 0
        while True:
            if animate_this_episode:
                env.render()
                time.sleep(0.1)
            obs.append(ob)
            #====================================================================================#
            #                           ----------PROBLEM 7. Sample trajectory----------
            #====================================================================================#
            # YOUR CODE HERE
            ac = self.sess.run(self.sy_sampled_ac, feed_dict={self.sy_ob_no : ob[None]})
            ac = ac[0]
            acs.append(ac)
            ob, rew, done, _ = env.step(ac)
            rewards.append(rew)
            steps += 1
            if done or steps > self.max_path_length:
                break
        path = {"observation" : np.array(obs, dtype=np.float32), 
                "reward" : np.array(rewards, dtype=np.float32), 
                "action" : np.array(acs, dtype=np.float32)}
        return path

    #====================================================================================#
    #                           ----------PROBLEM 8. Compute sum of rewards----------
    #====================================================================================#
    def sum_of_rewards(self, re_n):
        """
            Monte Carlo estimation of the Q function.

            let sum_of_path_lengths be the sum of the lengths of the paths sampled from 
                Agent.sample_trajectories
            let num_paths be the number of paths sampled from Agent.sample_trajectories

            arguments:
                re_n: length: num_paths. Each element in re_n is a numpy array 
                    containing the rewards for the particular path

            returns:
                q_n: shape: (sum_of_path_lengths). A single vector for the estimated q values 
                    whose length is the sum of the lengths of the paths

            ----------------------------------------------------------------------------------
            
            Your code should construct numpy arrays for Q-values which will in turn be fed to the placeholder you defined in 
            Agent.define_placeholders. 
            
            Recall that the expression for the policy gradient PG is
            
                  PG = E_{tau} [sum_{t=0}^T grad log pi(a_t|s_t) * Q_t]
            
            where 
            
                  tau=(s_0, a_0, ...) is a trajectory,
                  Q_t is the Q-value at time t, Q^{pi}(s_t, a_t).
            
            Instead of Q^{pi}(s_t, a_t), we use the total discounted reward summed over 
            entire trajectory (regardless of which time step the Q-value should be for). 
        
            For this case, the policy gradient estimator is
        
                E_{tau} [sum_{t=0}^T grad log pi(a_t|s_t) * Ret(tau)]
        
            where
        
                Ret(tau) = sum_{t'=0}^T gamma^t' r_{t'}.
        
            Thus, you should compute
        
                Q_t = Ret(tau)
            
            
            Store the Q-values for all timesteps and all trajectories in a variable 'q_n',
            like the 'ob_no' and 'ac_na' above. 
        """
        # YOUR CODE HERE
        q_n = []
        if self.reward_to_go:
            # TODO: Estimate the Q value Q^{pi}(s_t, a_t) using rewards from that entire trajectory
            # HINT1: value of each point (t) = total discounted reward summed over the entire trajectory (from 0 to T-1)
            # In other words, q(s_t, a_t) = sum_{t'=0}^{T-1} gamma^t' r_{t'}
            for re in re_n:                    
                q=np.zeros(len(re))
                q[-1]=re[-1]
                for i in reversed(range(len(re)-1)):
                    q[i]=re[i] + self.gamma * q[i+1]
                q_n.extend(q)

        else:
            # TODO: Estimate the Q value Q^{pi}(s_t, a_t) as the reward-to-go
            # HINT1: value of each point (t) = total discounted reward summed over the remainder of that trajectory (from t to T-1)
            # In other words, q(s_t, a_t) = sum_{t'=t}^{T-1} gamma^(t'-t) * r_{t'}
            for re in re_n:
                q = 0
                for r in reversed(re):
                    q *= self.gamma
                    q += r
                x = np.ones(shape=[len(re)]) * q
                q_n.extend(x)
        return q_n

    def estimate_return(self, ob_no, re_n):
        """
            Estimates the returns over a set of trajectories.

            let sum_of_path_lengths be the sum of the lengths of the paths sampled from 
                Agent.sample_trajectories
            let num_paths be the number of paths sampled from Agent.sample_trajectories

            arguments:
                ob_no: shape: (sum_of_path_lengths, ob_dim)
                re_n: length: num_paths. Each element in re_n is a numpy array 
                    containing the rewards for the particular path

            returns:
                adv_n: shape: (sum_of_path_lengths). A single vector for the estimated 
                    q values(not self.nn_baseline) / advantages(self.nn_baseline)
                    whose length is the sum of the lengths of the paths
        """
        # YOUR CODE HERE
        
        q_n = self.sum_of_rewards(re_n)
        adv = None
        if self.nn_baseline :
            b_n = self.sess.run(self.sy_bl_pred , feed_dict = {self.sy_ob_no: ob_no})
            b_n = np.mean(q_n) + b_n * np.std(q_n)
            adv = q_n - b_n
        return q_n, adv


    def update_parameters(self, ob_no, ac_na, q_n, targets_n):
        """ 
            Update the parameters of the policy and (possibly) the neural network baseline, 
            which is trained to approximate the value function.

            arguments:
                ob_no: shape: (sum_of_path_lengths, ob_dim)
                ac_na: shape: (sum_of_path_lengths).
                q_n: shape: (sum_of_path_lengths). A single vector for the estimated q values 
                    whose length is the sum of the lengths of the paths

            returns:
                nothing

        """

        #====================================================================================#
        #                           ----------PROBLEM 8. Update policy----------
        # Performing the Policy Update
        #====================================================================================#

        # Call the update operation necessary to perform the policy gradient update based on 
        # the current batch of rollouts.
        # 
        # For debug purposes, you may wish to save the value of the loss function before
        # and after an update, and then log them below. 

        # YOUR CODE HERE
        if self.nn_baseline:
            target_n = (q_n - np.mean(q_n)) / (np.std(q_n)+1e-8)
            self.sess.run([self.baseline_update_op], feed_dict={self.sy_targets_n: target_n, self.sy_ob_no: ob_no})
            self.sess.run([self.update_op], feed_dict={self.sy_ob_no: ob_no, self.sy_ac_na: ac_na, self.sy_q_n: targets_n} )
        else:
            self.sess.run([self.update_op], feed_dict={self.sy_ob_no: ob_no, self.sy_ac_na: ac_na, self.sy_q_n: q_n} )

def train_PG(
        exp_name,
        env_name,
        n_iter, 
        gamma, 
        min_timesteps_per_batch, 
        max_path_length,
        learning_rate,
        animate, 
        logdir,
        seed,
        n_layers,
        size,
        rtg,
        nn_baseline):

    start = time.time()

    #========================================================================================#
    # Set Up Logger
    #========================================================================================#
    setup_logger(logdir, locals())

    #========================================================================================#
    # Set Up Env
    #========================================================================================#

    # Make the gym environment
    env = gym.make(env_name)
    env = gym.wrappers.Monitor(env, osp.join(logdir, 'video'), force=True)

    # Set random seeds
    tf.set_random_seed(seed)
    np.random.seed(seed)
    env.seed(seed)

    # Maximum length for episodes
    max_path_length = max_path_length or env.spec.max_episode_steps

    # Is this env continuous, or self.discrete?
    discrete = isinstance(env.action_space, gym.spaces.Discrete)
    assert discrete == True

    # Observation and action sizes
    ob_dim = env.observation_space.shape[0]
    ac_dim = env.action_space.n

    #========================================================================================#
    # Initialize Agent
    #========================================================================================#
    computation_graph_args = {
        'n_layers': n_layers,
        'ob_dim': ob_dim,
        'ac_dim': ac_dim,
        'size': size,
        'learning_rate': learning_rate,
        }

    sample_trajectory_args = {
        'animate': animate,
        'max_path_length': max_path_length,
        'min_timesteps_per_batch': min_timesteps_per_batch,
    }

    estimate_return_args = {
        'gamma': gamma,
    }

    agent_args = {
        'nn_baseline': nn_baseline,
        'reward_to_go': rtg,
    }

    agent = Agent(computation_graph_args, sample_trajectory_args, estimate_return_args, agent_args
        )

    # build computation graph
    agent.build_computation_graph()

    # tensorflow: config, session, variable initialization
    agent.init_tf_sess()

    #========================================================================================#
    # Training Loop
    #========================================================================================#

    total_timesteps = 0
    for itr in range(n_iter):
        print("********** Iteration %i ************"%itr)
        paths, timesteps_this_batch = agent.sample_trajectories(itr, env)
        total_timesteps += timesteps_this_batch

        # Build arrays for observation, action for the policy gradient update by concatenating 
        # across paths
        ob_no = np.concatenate([path["observation"] for path in paths])
        ac_na = np.concatenate([path["action"] for path in paths])
        re_n = [path["reward"] for path in paths]

        q_n, adv = agent.estimate_return(ob_no, re_n)

        agent.update_parameters(ob_no, ac_na, q_n, adv)
        
        # Log diagnostics
        returns = [path["reward"].sum() for path in paths]
        ep_lengths = [pathlength(path) for path in paths]
        logz.log_tabular("Time", time.time() - start)
        logz.log_tabular("Iteration", itr)
        logz.log_tabular("mean reward", np.mean(returns))
        logz.log_tabular("std reward", np.std(returns))
        logz.log_tabular("max reward", np.max(returns))
        logz.log_tabular("min reward", np.min(returns))
        logz.log_tabular("EpLenMean", np.mean(ep_lengths))
        logz.log_tabular("EpLenStd", np.std(ep_lengths))
        logz.log_tabular("TimestepsThisBatch", timesteps_this_batch)
        logz.log_tabular("TimestepsSoFar", total_timesteps)
        logz.dump_tabular()
        logz.pickle_tf_vars()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='CartPole-v0')
    parser.add_argument('--exp_name', type=str, default='vpg')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--discount', type=float, default=1.0)
    parser.add_argument('--n_iter', '-n', type=int, default=50)
    parser.add_argument('--batch_size', '-b', type=int, default=10000)
    parser.add_argument('--ep_len', '-ep', type=float, default=-1.)
    parser.add_argument('--learning_rate', '-lr', type=float, default=5e-3)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--n_experiments', '-e', type=int, default=1)
    parser.add_argument('--n_layers', '-l', type=int, default=2)
    parser.add_argument('--size', '-s', type=int, default=64)
    parser.add_argument('--reward_to_go', '-rtg', action='store_true')
    parser.add_argument('--nn_baseline', action='store_true')
    args = parser.parse_args()

    logdir = args.exp_name + '_' + args.env_name + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
    print("Result dir:", logdir)
    if not(os.path.exists(logdir)):
        os.makedirs(logdir)

    max_path_length = args.ep_len if args.ep_len > 0 else None

    processes = []

    for e in range(args.n_experiments):
        seed = args.seed + 10*e
        print('Running experiment with seed %d'%seed)

        def train_func():
            train_PG(
                exp_name=args.exp_name,
                env_name=args.env_name,
                n_iter=args.n_iter,
                gamma=args.discount,
                min_timesteps_per_batch=args.batch_size,
                max_path_length=max_path_length,
                learning_rate=args.learning_rate,
                animate=args.render,
                logdir=os.path.join(logdir,'%d'%seed),
                seed=seed,
                n_layers=args.n_layers,
                size=args.size,
                rtg=args.reward_to_go,
                nn_baseline=args.nn_baseline,
                )
        # # Awkward hacky process runs, because Tensorflow does not like
        # # repeatedly calling train_PG in the same thread.
        p = Process(target=train_func, args=tuple())
        p.start()
        processes.append(p)
        # if you comment in the line below, then the loop will block 
        # until this process finishes
        # p.join()

    for p in processes:
        p.join()

if __name__ == "__main__":
    main()

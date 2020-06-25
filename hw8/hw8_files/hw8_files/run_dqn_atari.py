import argparse
import gym
from gym import wrappers
import os.path as osp
import random
import uuid
import datetime
import dateutil.tz
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

import dqn
from dqn_utils import *
from atari_wrappers import *
import logz

def lander_model(img_in, num_actions, scope, reuse=False):
    with tf.variable_scope(scope, reuse=reuse):
        out = img_in
        with tf.variable_scope("action_value"):
            out = tf.layers.dense(out, units=64, activation=tf.nn.relu)
            out = tf.layers.dense(out, units=64, activation=tf.nn.relu)
            out = tf.layers.dense(out, units=num_actions, activation=None)
    
        return out

def atari_model(img_in, num_actions, scope, reuse=False):
    # as described in https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf
    with tf.variable_scope(scope, reuse=reuse):
        out = img_in
        with tf.variable_scope("convnet"):
            # original architecture
            out = tf.layers.conv2d(out, units=32, kernel_size=8, stride=4, activation=tf.nn.relu)
            out = tf.layers.conv2d(out, units=64, kernel_size=4, stride=2, activation=tf.nn.relu)
            out = tf.layers.conv2d(out, units=64, kernel_size=3, stride=1, activation=tf.nn.relu)
        out = tf.layers.flatten(out)
        with tf.variable_scope("action_value"):
            out = tf.layers.dense(out, units=512,         activation=tf.nn.relu)
            out = tf.layers.dense(out, units=num_actions, activation=None)

        return out

def atari_learn(env,
                session,
                num_timesteps,
                result_dir):
    # This is just a rough estimate
    num_iterations = float(num_timesteps) / 4.0

    lr_multiplier = 1.0
    lr_schedule = PiecewiseSchedule([
                                         (0,                   1e-4 * lr_multiplier),
                                         (num_iterations / 10, 1e-4 * lr_multiplier),
                                         (num_iterations / 2,  5e-5 * lr_multiplier),
                                    ],
                                    outside_value=5e-5 * lr_multiplier)
    lander_optimizer = dqn.OptimizerSpec(
        constructor=tf.train.AdamOptimizer,
        kwargs={},
        lr_schedule=ConstantSchedule(1e-3)
    )

    def stopping_criterion(env, t):
        # notice that here t is the number of steps of the wrapped env,
        # which is different from the number of steps in the underlying env
        return get_wrapper_by_name(env, "Monitor").get_total_steps() >= num_timesteps

    exploration_schedule = PiecewiseSchedule(
        [
            (0, 1),
            (num_timesteps*0.1, 0.02),
        ], outside_value=0.02
    )

    dqn.learn(
        env=env,
        q_func=lander_model,
        optimizer_spec=lander_optimizer,
        session=session,
        exploration=exploration_schedule,
        stopping_criterion=stopping_criterion,
        replay_buffer_size=50000,
        batch_size=32,
        gamma=1,
        learning_starts=1000,
        learning_freq=1,
        frame_history_len=1,
        target_update_freq=3000,
        grad_norm_clipping=10,
        lander=True,
        rew_file=osp.join(result_dir, 'episode_rewards.pkl'),
    )
    env.close()

def get_available_gpus():
    from tensorflow.python.client import device_lib
    local_device_protos = device_lib.list_local_devices()
    return [x.physical_device_desc for x in local_device_protos if x.device_type == 'GPU']

def set_global_seeds(i):
    try:
        import tensorflow.compat.v1 as tf
    except ImportError:
        pass
    else:
        tf.set_random_seed(i)
    np.random.seed(i)
    random.seed(i)

def get_session():
    tf.reset_default_graph()
    tf_config = tf.ConfigProto(
        inter_op_parallelism_threads=1,
        intra_op_parallelism_threads=1)
    session = tf.Session(config=tf_config)
    print("AVAILABLE GPUS: ", get_available_gpus())
    return session

def get_env(task, seed, logdir):
    env = gym.make('LunarLander-v2')

    set_global_seeds(seed)
    env.seed(seed)

    env = wrappers.Monitor(env, osp.join(logdir, "video"), force=True)
    #env = wrap_deepmind(env)

    return env

def main():
    # Get Atari games.
    task = gym.make('LunarLander-v2')

    file_dir = osp.dirname(osp.abspath(__file__))
    unique_name = datetime.datetime.now(dateutil.tz.tzlocal()).strftime('%Y_%m_%d_%H_%M_%S_%f_%Z') + '__' + str(uuid.uuid4())
    result_dir = osp.join(file_dir, unique_name)

    logz.configure_output_dir(result_dir)
    logz.save_params(dict(
            exp_name=unique_name,
    ))

    # Run training
    seed = 1
    print('random seed = %d' % seed)
    env = get_env(task, seed, result_dir)
    session = get_session()
    atari_learn(env, session, num_timesteps=5e5, result_dir=result_dir)

if __name__ == "__main__":
    main()

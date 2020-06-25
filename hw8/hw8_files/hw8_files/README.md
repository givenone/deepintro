# Introduction to Deep Learning HW 6: Deep Q-Learning and Vanilla Policy Gradient

Your docker instance is all set-up for this homework.

## Deep Q-Learning

The only file that you need to modify is `dqn.py`.

To run experiment, execute `./run_dqn.sh`. It creates a new log directory with a name formatted like `2018_12_05_04_52_28_424073_KST__12dca7b8-270b-4f8f-a038-e7379fa57be4`.

To create a learning curve plot, run `plot.py` passing the log directory as an argument, like `python3.5 plot.py 2018_12_05_04_52_28_424073_KST__12dca7b8-270b-4f8f-a038-e7379fa57be4`.

See the given PDF for further instructions.

The starter code was based on an implementation of Q-learning for Atari generously provided by Szymon Sidor from OpenAI.

## Vanilla Policy Gradient

The only file that you need to modify is `train_pg_f18.py`.

To run experiment, execute `./run_vpg.sh`. It creates a new log directory with a name formatted like `vpg_CartPole-v0_01-06-2019_06-18-48`.

To create a learning curve plot, run `plot.py` passing the log directory as an argument, like `python3.5 plot.py vpg_CartPole-v0_01-06-2019_06-18-48 --plot_vpg --value="mean reard"`.

The starter code was based on an implementation of Vanilla Policy Gradient generously provided by John Schulman from OpenAI.
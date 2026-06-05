# SHAC++ integration with videogame simulators

## Problem

The problem was to achieve a good result from the Deep Reinforcement Learning model called SHAC++ in the simulated environment of [Breakout](https://ale.farama.org/environments/breakout/).
The maximum of the reward from Breakout is 864, so a reward of 600/700 could be considered already good. Another way to see if the model actually learnt to play Breakout is to view an actual play of the model on the game.
For all of the experiments SHAC++ is used with a single agent.

For second, third and fourth experiments there is a common abstract class that uses the template method pattern to adhere to the DRY principle. The abstract class is [here](./src/environments/scenarios/BaseGymScenario.py)

## First experiment

The first experiment to be done uses a pre-trained CNN that follows the architecture used by DQN (Mnih et al., 2015) to extract futures from the pixels of the game, found [here](src/environments/scenarios/Breakout.py)

The training was done with the Transformer model.

### Observations

The features extracted by the CNN are used as the observations of the world.

### Actions

The SHAC++ agent outputs actions in a range of [-1, 1], while the Breakout environment accepts only discrete actions like 0 for NO-OP, 2 for RIGHT and 3 for LEFT, so a discretization of the agent action is needed.

The discretization choiced is very simple:
- agent action in range [-1, -0,33) -> LEFT
- agent action in range [-0,33, 0,33] -> NO-OP
- agent action in range (0.33, 1] -> RIGHT
  
In this way every action should have the same probability of being chosen.

### Results

The SHAC++ model seems to learn well the world from these graphs:
- [World accuracy](./graphs/breakout/transformer/42/breakout_world_accuracy.pdf)
- [World loss](./graphs/breakout/transformer/42/breakout_world_loss.pdf)

The reward instead is not learnt well:
- [Reward accuracy](./graphs/breakout/transformer/42/breakout_reward_accuracy.pdf)
- [Reward loss](./graphs/breakout/transformer/42/breakout_reward_loss.pdf)

Also, while watching it play Breakout, it seems to just move randomly on the screen.

## Second experiment

To better understand the problem behind the model that didn't trained well on Breakout with CNN, the second experiment consisted of training it on Breakout RAM, a version of Breakout that instead of using pixels as observation of the world, uses the ram of the game (like the ball and paddle positions). This can help also performance because it doesn't need a CNN.

The experiment is found [here](./src/environments/scenarios/Breakout_Ram.py)

Using transformer instead of mlp as model for the Neural Networks does not change the relevant outcome of the model training.

### Observations

The Ram is used as the observation of the world, normalized to [0, 1] for the neural network.
It is also important to know these two values of the RAM:
- Paddle X position: RAM[72] (range ~40-168)
- Ball X position: RAM[99] (range ~0-160)

### Actions

Two continuos actions are taken by the agent and then they are converted to RIGHT or LEFT via argmax (). Differently from Breakout, NO-OP is not used to have better performance (moreover a no operation action is not actually needed).

Before trying this discretization, also other discretization algorithm were tried, like the one used in the Breakout with CNN.

### Reward modifications

Having the RAM as the state of the world can help also to change the rewards of the environment, to give the model a reward at every step, and various options were tested:
- Tracking bonus: The more the X of the paddle (RAM[72]) is closer to the X of the ball (RAM[99]), the higher the reward.
- Interception bonus: Positive reward for moving the paddle toward the ball position, negative otherwise.

In each case tested, the training of the model didn't seem to be affected enough to change the actual result.

### Results

The SHAC++ model seems to learn well the world also in this experiment from these graphs:
- [World accuracy](./graphs/breakout_ram/mlp/43/breakout-world-accuracy.pdf)
- [World loss](./graphs/breakout_ram/mlp/43/breakout-world-loss.pdf)

The reward accuracy seems to get an high value, but it is not consistent and the loss also shows an unconsistent behaviour:
- [Reward accuracy](./graphs/breakout_ram/mlp/43/breakout-reward-accuracy.pdf)
- [Reward loss](./graphs/breakout_ram/mlp/43/breakout-reward-loss.pdf)

As for the actual play of the game, it seems to not behave much differently from the Brakout with CNN scenario. With some change of the bonuses of the reward and the model of the neural network, it seems that the paddle at first goes left, take the ball one time then goes right and stays there, losing the game.

The model of the neural networks seems the first to impact this, making MLP having the behaviour described, while the transformer seems to have a random behaviour.

## Third experiment

Not being able to understand if the problem in the second experiment is the discretization of the actions, the reward not being "dense" (meaning it is not changing at every step of the training) or something else, a third experiment has been done, trying to train SHAC++ to play [Pendulum](https://gymnasium.farama.org/environments/classic_control/pendulum/).

The experiment can be found [here](./src/environments/scenarios/Pendulum.py).

There is not real difference between transformer and mlp.

### Observations

The observations used are ram bytes of the game, like the angular velocity of the pendulum and its position.

### Actions

One continuos action is taken from the output of the SHAC++ model and then scaled to adapt to the continuos action of the environment, so it will be in a range of [-2, 2], representing the torque applied to the pendulum.

### Results

The SHAC++ model seems to learn well the world also in this experiment from these graphs:
- [World accuracy](./graphs/pendulum/mlp/42/pendulum-world-accuracy.pdf)
- [World loss](./graphs/pendulum/mlp/42/pendulum-world-loss.pdf)

The reward accuracy and loss have good values too, being also consistent in time:
- [Reward accuracy](./graphs/pendulum/mlp/42/pendulum-reward-accuracy.pdf)
- [Reward loss](./graphs/pendulum/mlp/42/pendulum-reward-loss.pdf)

Also in the actual play of the game it can be seen that the model can actually play the game with a good reward (the pendulum is for almost all the time in the up position)

## Fourth experiment

As the last experiment has been tried to let SHAC++ play [Cartpole](https://gymnasium.farama.org/environments/classic_control/cart_pole/).
Pendulm has a dense reward and a continuos action, while cartpole has a dense reward but without continuous actions.

The experiment can be found [here](./src/environments/scenarios/CartPole.py).

There is not real difference between transformer and mlp.

### Observations

The observations used are RAM bytes of the game, in particular the cart position, cart velocity, pole angle and pole angular velocity.

### Actions

There are two discrete actions for the game, and the same discretization used in the second experiment was tried.

### Results

The SHAC++ model seems to not being able to learn well the world in this experiment:
- [World accuracy](./graphs/cartpole/mlp/42/cartpole_world_accuracy.pdf)
- [World loss](./graphs/cartpole/mlp/42/cartpole_world_loss.pdf)

The reward accuracy and loss have good values instead, being also consistent in time:
- [Reward accuracy](./graphs/cartpole/mlp/42/cartpole_reward_accuracy.pdf)
- [Reward loss](./graphs/cartpole/mlp/42/cartpole_reward_loss.pdf)

As for the actual play, the model didn't seem to be able to play it well, making fall the pole on the cart interrupting really early the game.

# This GitHub repository was used as a reference in tandem with my previous work: https://github.com/johnnycode8/gym_solutions/blob/main/mountain_car_dql.py
import sys
import time

import gymnasium as gym
import numpy as np
from collections import deque
import random
import torch
from torch import nn
import torch.nn.functional
import matplotlib.pyplot as plt

# Define model
class BuildDQNModel(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()

        # Define network layers
        self.input_layer = nn.Linear(state_size, 8) # NN has one hidden layer; 4 neurons seems to be the ideal parameter. Greater than this prevents
                                                                # the agent from learning consistently.
        self.output_layer = nn.Linear(8, action_size)

    def forward(self, x):
        x = torch.nn.functional.relu(self.input_layer(x))  # Apply ReLU activation
        x = self.output_layer(x)
        return x


# Define memory for Experience Replay
class ExperienceReplay:
    def __init__(self, MEMORY_SIZE):
        self.memory = deque(maxlen=MEMORY_SIZE)

    def append(self, transition):
        self.memory.append(transition)

    def sample(self, sample_size):
        return random.sample(self.memory, sample_size)

    def __len__(self):
        return len(self.memory)

class MountainCar:
    ALPHA = 0.001  # learning rate; .001 works well with the current batch size of 64
    GAMMA = 0.9  # discount rate
    EPSILON = 1.0 # EPSILON factor
    TARGET_UPDATE_FREQ = 100000
    MEMORY_SIZE = 100000
    BATCH_SIZE = 64 # this batch size, combined with the learning rate above, seem to provide the most consistent results

    # Neural network features
    loss_fn = nn.HuberLoss()  # Huber Loss provides better results compared to other loss functions
    optimizer = None

    losses = []

    # Train the environment
    def train(self, EPISODES):
        try:
            # Environment setup
            env = gym.make('MountainCar-v0')


        except Exception:
            print ('Failed to initialize environment.')
            sys.exit(1)

        try:
            # Experience Replay setup
            memory = ExperienceReplay(self.MEMORY_SIZE)
        except Exception:
            print ('Failed to initialize experience replay object.')
            sys.exit(1)

        state_size = env.observation_space.shape[0]
        action_size = env.action_space.n

        try:
            # Create models and set weights
            policy_model = BuildDQNModel(state_size, action_size)
            target_model = BuildDQNModel(state_size, action_size)

            target_model.load_state_dict(policy_model.state_dict())
        except Exception:
            print ('Failed to initialize neural networks.')
            sys.exit(1)

        self.optimizer = torch.optim.Adam(policy_model.parameters(), lr=self.ALPHA) # Optimizer Adam works best with the current neural network
                                                                                    # I had tried both SGD and AdamW, but they failed to converge

        rewards_per_episode = []

        step_count = 0
        goal_reached = False
        best_rewards = -200

        episode_timer = time.perf_counter()
        for e in range(EPISODES):
            state = env.reset()[0]
            terminated = False  # True when agent reaches goal
            total_reward = 0

            # Agent takes actions until goal is met or 1000 steps is exceeded
            # I had to override the maximum steps the agent could take so that it would be able to find the solution. 200 steps is not enough time for
            # the car to reach the goal via exploration
            while not terminated and total_reward > -1000:

                if random.random() < self.EPSILON:
                    # Exploration action
                    action = env.action_space.sample()
                else:
                    # Exploitation action
                    with torch.no_grad(): # Using torch.no_grad() seems to have little effect on the training results but speeds up training time
                        action = policy_model(torch.FloatTensor(state)).argmax().item()

                next_state, reward, terminated, truncated, _ = env.step(action)

                memory.append((state, action, next_state, reward, terminated))

                state = next_state

                total_reward += reward
                step_count += 1

            rewards_per_episode.append(total_reward)
            if (terminated):
                goal_reached = True # Goal was reached for the first time

            if e != 0 and e % 1000 == 0:
                avg_reward = np.mean(rewards_per_episode[-1000:])
                elapsed_time = time.perf_counter() - episode_timer
                print(f'Episode {e}: Exploration: {self.EPSILON}, Average Reward: {avg_reward}, Elapsed Time: {elapsed_time}')

            # Save policy of best run
            if total_reward > best_rewards:
                best_rewards = total_reward
                print(f'Best rewards so far: {best_rewards}')
                torch.save(policy_model.state_dict(), f"mountaincar_dql_{e}.pt")

            # This branch opens whenever the agent reaches the top of the hill once. It is responsible for optimizing
            # the models and updating the target model weights
            if len(memory) > self.BATCH_SIZE and goal_reached:
                batch = memory.sample(self.BATCH_SIZE)
                self.optimize(batch, policy_model, target_model)

                self.EPSILON = max(self.EPSILON - 1 / EPISODES, 0) # Decay exploration

                # Update target model weights
                if step_count > self.TARGET_UPDATE_FREQ:
                    target_model.load_state_dict(policy_model.state_dict())
                    step_count = 0

        env.close()
        plt.plot(self.losses)
        plt.xlabel('Training Episodes')
        plt.ylabel('Loss')
        plt.title('Training loss')
        plt.show()

    # Optimize policy network
    def optimize(self, batch, policy_model, target_model):

        try:
            current_q_list = [] # Stores current states
            target_q_list = [] # Stores states with updated q values

            for (state, action, next_state, reward, terminal) in batch:
                current_q = policy_model(torch.FloatTensor(state))
                current_q_list.append(current_q)

                target_q = target_model(torch.FloatTensor(state))
                next_q = target_model(torch.FloatTensor(next_state))

                # Calculate target q value and adjust the action
                with torch.no_grad():
                    if terminal:
                        target_q[action] = torch.FloatTensor([reward])
                    else:
                        target_q[action] = reward + self.GAMMA * next_q.max()

                target_q_list.append(target_q)

            # Compute loss for the whole batch
            loss = self.loss_fn(torch.stack(current_q_list), torch.stack(target_q_list))
            self.losses.append(loss.item())

            # Optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        except Exception:
            print ('Failed to optimize neural network.')
            sys.exit(1)

    def test(self, EPISODES, weights):

        try:
            env = gym.make("MountainCar-v0", render_mode="human")
            state_size = env.observation_space.shape[0]
            action_size = env.action_space.n

            model = BuildDQNModel(state_size, action_size)
            model.load_state_dict(torch.load(weights))
        except Exception:
            print ('Failed to initialize test environment.')
            sys.exit(1)

        for e in range(EPISODES):
            state, _ = env.reset()
            done = False
            total_reward = 0

            while not done:
                action = model(torch.FloatTensor(state)).argmax().item()

                next_state, reward, terminated, truncated, _ = env.step(action)

                state = next_state

                total_reward += reward
                done = terminated or truncated
            print (f"Episode {e + 1}: Total Reward = {total_reward:.2f}")

        env.close()

mountaincar = MountainCar()
#mountaincar.train(20000)
mountaincar.test(10, "mountaincar_dql_18638.pt")
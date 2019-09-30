from model import Model
import numpy as np
import tensorflow as tf
import gym
from gym.wrappers.monitoring.video_recorder import VideoRecorder
from gym.wrappers.monitor import Monitor

# Test du modèle
env = gym.make("CartPole-v0")
env = Monitor(env, "videos",force=True)
model = Model(num_actions=env.action_space.n)

obs = env.reset()
action, value = model.action_value(obs[None, :])
print(action, value)

# Création de l'agent 
class Agent():
    def __init__(self, model):
        self.params = { "value": 0.5,
                        "entropy": 0.0001,
                        "gamma": 0.99}

        self.model = model 
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(lr=0.0005),
            loss = [self._logits_loss, self._value_loss]
        )

    def test(self, env, render=True):
        obs, done, ep_reward = env.reset(), False, 0
        while not done:

            action, _ = self.model.action_value(obs[None, :])
            obs, reward, done, _ = env.step(action)
            ep_reward += reward

            if render:
                env.render()
        return ep_reward

    def _value_loss(self, returns, value):
        return self.params["value"]*tf.keras.losses.mean_squared_error(returns, value)
    
    def _logits_loss(self, acts_and_advs, logits):
        actions, advantages = tf.split(acts_and_advs, 2, axis=-1)

        weighted_sparse_ce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        actions = tf.cast(actions, tf.int32)

        policy_loss = weighted_sparse_ce(actions, logits, sample_weight=advantages)
        entropy_loss = tf.keras.losses.categorical_crossentropy(logits, logits, from_logits=True)

        return policy_loss - self.params["entropy"]*entropy_loss

    def train(self, env, batch_size=32, updates=1000):

        actions= np.empty((batch_size, ), dtype=np.int32)
        rewards, dones, values = np.empty((3, batch_size))
        observations = np.empty((batch_size, ) + env.observation_space.shape)

        ep_rews = [0.0]
        next_obs = env.reset()

        for update in range(updates):
            for step in range(batch_size):
                
                observations[step] = next_obs.copy()
                actions[step], values[step] = self.model.action_value(next_obs[None, :])
                next_obs, rewards[step], dones[step], _ = env.step(actions[step])

                ep_rews[-1] += rewards[step]
                if dones[step]:
                    ep_rews.append(0.0)
                    next_obs = env.reset()
                
            _, next_value = self.model.action_value(next_obs[None, :])
            returns, advs = self._returns_advantages(rewards, dones, values, next_value)

            acts_and_advs = np.concatenate([actions[: , None], advs[:, None]], axis=-1)

            losses = self.model.train_on_batch(observations, [acts_and_advs, returns])
            
            print(update, ep_rews[-1])
        return ep_rews

    def _returns_advantages(self, rewards, dones, values, next_value):

        returns = np.append(np.zeros_like(rewards), next_value, axis=-1)

        for t in reversed(range(rewards.shape[0])):
            returns[t] = rewards[t] + self.params["gamma"] * returns[t+1] * (1-dones[t])
        returns = returns[:-1]

        advantages = returns - values
        return returns, advantages


agent = Agent(model)

# Test de l'agent sur un modèle aléatoire
"""
rewards_sum = agent.test(env)
print("%d out of 200" % rewards_sum)
"""

# Entrainement de l'agent 
rewards_history = agent.train(env, 32, 250)
print("Finished Training, Testing...")
env.close()

# Test de l'entrainement
print("{} out of 200".format(agent.test(env)))
env.close()



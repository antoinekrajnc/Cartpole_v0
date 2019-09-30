import numpy as np
import tensorflow as tf
tf.__version__


# Retourne des actions aléatoires en fonction de probabilités
class ProbabilityDistribution(tf.keras.Model):
  def call(self, logits):
    return tf.squeeze(tf.random.categorical(logits, 1), axis=1)


# Création du modèle 
class Model(tf.keras.Model):
  def __init__(self, num_actions):
    super().__init__("mlp_policy")

    self.hidden_1_logs = tf.keras.layers.Dense(128, activation="relu")
    self.hidden_1_val = tf.keras.layers.Dense(128, activation="relu")

    self.value = tf.keras.layers.Dense(1, name="value")

    self.logits = tf.keras.layers.Dense(num_actions, name="policy_logits")
    self.softmax = tf.keras.layers.Softmax()

  def call(self, inputs):
    
    x = tf.convert_to_tensor(inputs, dtype=tf.float32)

    hidden_logs = self.hidden_1_logs(x)
    hidden_vals = self.hidden_1_val(x)

    return self.logits(hidden_logs), self.value(hidden_vals)

  def action_value(self, obs):

    logits, value = self.predict(obs)
    action = self.softmax(logits)

    return np.argmax(np.squeeze(action, axis=-1)), np.squeeze(value, axis=-1)
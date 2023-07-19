import random

import chess
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import LSTM
from keras.optimizers.legacy import Adam
from collections import deque

from keras.src.layers import TimeDistributed

POSSIBLE_SENSES = [i for i in range(64) if 1 <= (i // 8) <= 6 and 1 <= (i % 8) <= 6]

class RLAgent:

    def __init__(self, state_size=8 * 8 * 12, action_size=64 * 64, alpha=0.001, gamma=0.9, epsilon=0.1,
                 memory_size=10000):
        self.state = None
        self.state_size = state_size
        self.action_size = action_size
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.memory = deque(maxlen=memory_size)
        self.move_model = self.build_move_model()  # Model for move actions
        self.sense_model = self.build_sense_model()  # Model for sense actions
        self.target_move_model = self.build_move_model()  # Target model for move actions
        self.target_sense_model = self.build_sense_model()  # Target model for sense actions
        self.update_target_model()

    def build_move_model(self):
        model = Sequential()
        model.add(TimeDistributed(Flatten(), input_shape=(None, 8, 8, 13)))  # Flatten each board
        model.add(LSTM(24, return_sequences=False))  # LSTM layer
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.alpha))
        return model

    def build_sense_model(self):
        model = Sequential()
        model.add(TimeDistributed(Flatten(), input_shape=(None, 8, 8, 13)))  # Flatten each board
        model.add(LSTM(24, return_sequences=False))  # LSTM layer
        model.add(Dense(24, activation='relu'))
        model.add(Dense(36, activation='linear'))  # Only 36 output values for the inner 6x6 squares
        model.compile(loss='mse', optimizer=Adam(lr=self.alpha))
        return model

    def update_target_model(self):
        self.target_move_model.set_weights(self.move_model.get_weights())
        self.target_sense_model.set_weights(self.sense_model.get_weights())

    def reset(self):
        self.state = None
        self.action = None

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))  # Store the experience in the memory

    def get_q_values(self, state):
        return self.model.predict(state)  # Use the neural network to get the Q-values

    def choose_move(self, state, possible_moves):
        self.state = state
        if random.random() < self.epsilon:  # Exploration
            self.action = random.choice(possible_moves)
        else:  # Exploitation
            state_batch = np.expand_dims(state, axis=0)  # Add batch dimension
            q_values = self.move_model.predict(state_batch)
            mask = np.zeros(self.action_size, dtype=bool)
            for move in possible_moves:
                action = move_to_action(move)
                mask[action] = True
            masked_q_values = np.where(mask, q_values, -np.inf)
            action = np.argmax(masked_q_values)  # Choose the valid action with the highest Q-value
            self.action = action_to_move(action)
        return self.action

    def choose_sense(self, state):
        self.state = state
        if random.random() < self.epsilon:  # Exploration
            sense_square = random.choice(POSSIBLE_SENSES)
        else:  # Exploitation
            state_batch = np.expand_dims(state, axis=0)  # Add batch dimension
            q_values = self.sense_model.predict(state_batch)
            sense_square = POSSIBLE_SENSES[np.argmax(q_values)]  # Choose the valid action with the highest Q-value
        return sense_square

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)  # Sample a batch of experiences from the memory
        states, actions, rewards, next_states, dones = zip(*minibatch)
        states = np.array(states)
        next_states = np.array(next_states)
        move_q_values = self.move_model.predict(states)
        sense_q_values = self.sense_model.predict(states)
        next_move_q_values = self.target_move_model.predict(next_states)
        next_sense_q_values = self.target_sense_model.predict(next_states)
        for i, (state, action, reward, next_state, done) in enumerate(minibatch):
            target = reward
            if not done:
                target += self.gamma * np.amax(
                    next_move_q_values[i])  # Use the target network to compute the target Q-value
                target += self.gamma * np.amax(
                    next_sense_q_values[i])  # Use the target network to compute the target Q-value
            move_q_values[i][action] = target
            sense_q_values[i][action] = target
        self.move_model.fit(states, move_q_values, epochs=1, verbose=0)
        self.sense_model.fit(states, sense_q_values, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay  # Gradually decrease the exploration rate

    def train(self, batch_size):
        if len(self.memory) < batch_size:
            return

        samples = random.sample(self.memory, batch_size)
        for sample in samples:
            state, action, reward, new_state, done = sample
            target_move = self.target_move_model.predict(state)
            target_sense = self.target_sense_model.predict(state)
            if done:
                target_move[0][action] = reward
                target_sense[0][action] = reward
            else:
                Q_future_move = max(self.target_move_model.predict(new_state)[0])
                Q_future_sense = max(self.target_sense_model.predict(new_state)[0])
                target_move[0][action] = reward + Q_future_move * self.gamma
                target_sense[0][action] = reward + Q_future_sense * self.gamma
            self.move_model.fit(state, target_move, epochs=1, verbose=0)
            self.sense_model.fit(state, target_sense, epochs=1, verbose=0)

    def load(self, name):
        self.move_model.load_weights(f'{name}_move')
        self.sense_model.load_weights(f'{name}_sense')

    def save(self, name):
        self.move_model.save_weights(f'{name}_move')
        self.sense_model.save_weights(f'{name}_sense')




def action_to_move(action):
    from_square = action // 64
    to_square = action % 64
    move = chess.Move(from_square, to_square)
    return move

def move_to_action(move):
    action = move.from_square * 64 + move.to_square
    return action


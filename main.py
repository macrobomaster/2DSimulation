from rmaics import rmaics
import numpy as np
import random
import time
from keras.models import Model
from keras.layers import Input, Dense, LSTM, Attention
from keras.optimizers import Adam

# Define the game environment and its parameters
game = rmaics(agent_num=1, render=True)
state_dim = 23
num_actions = 8  # Number of possible actions

# Define the Transformer model
def build_transformer_model(state_dim, num_actions):
    inputs = Input(shape=(state_dim, 1))
    
    # LSTM layer with attention mechanism
    lstm_out = LSTM(64, return_sequences=True)(inputs)
    attn_out = Attention()([lstm_out, lstm_out])  # Add attention mechanism
    
    # Second LSTM layer with attention mechanism
    lstm_out2 = LSTM(64)(attn_out)
    attn_out2 = Attention()([lstm_out2, lstm_out2])  # Add attention mechanism
    
    # Output layer
    outputs = Dense(num_actions, activation='softmax')(attn_out2)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model

# Build and compile the model
model = build_transformer_model(state_dim, num_actions)
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001))

# Function to preprocess observations
def deal_with_states(observations):
    agent_states = np.array(observations['agent_states']).flatten()
    competition_info = np.array(observations['competition_info']).flatten()
    visible_agents = np.array(observations['visible_agents']).flatten() or np.array([])
    visible_enemies = np.array(observations['visible_enemies']).flatten() or np.array([])

    return np.concatenate((agent_states, competition_info, visible_agents, visible_enemies)).reshape(-1, 23, 1)

# Training loop
total_reward = 0
epsilon = 1.0  # Exploration rate
obs = game.reset()
training_duration = 10000  # Total training steps
episode_duration = 1000  # Number of steps per episode

for _ in range(training_duration // episode_duration):
    obs = game.reset()  # Reset the environment for each episode
    total_reward = 0
    
    for _ in range(episode_duration):
        state = deal_with_states(obs)
        
        # Exploration-exploitation trade-off
        if random.random() < epsilon:
            actions = np.random.randint(2, size=num_actions)  # Random action
        else:
            # Choose action based on model prediction
            actions_probs = model.predict(state)[0]
            chosen_action = np.random.choice(range(num_actions), p=actions_probs)
            actions = np.eye(num_actions)[chosen_action]  # One-hot encode chosen action
        
        # Execute action in the environment
        actions = np.array([actions])
        obs, reward, done, _ = game.step(actions=actions)
        total_reward += reward
        
        # Print total reward for monitoring
        print("Total reward:", total_reward)
        time.sleep(0.1)

        # Update the model
        state = deal_with_states(obs)
        target = reward  # For simplicity, since we don't have explicit future reward in this setting
        target_full = np.zeros((1, num_actions))
        target_full[0, np.argmax(actions)] = target  # Update target for chosen action
        model.fit(state, target_full, epochs=1, verbose=0)
        if done:
            break
        if total_reward < -1000:
            break
        epsilon *= 0.995  # Decaying exploration rate
        epsilon = max(0.01, epsilon)  # Ensure epsilon doesn't go below 0.01

import environment
import model
import torch
from itertools import count
from settings import DEVICE, SCREEN_WIDTH, TARGET_UPDATE, EPOCHS, EPSILON_END, EPSILON_START, EPSILON_DECAY
import time
import math
import cv2

# Environment Setup
world = environment.init()
world.reset()

# Log how long our agent lasts for each iteration
durations = []
episode_rewards = []


# Training
agent = model.Agent(DEVICE)

def normalize(screen):
    return screen / 255.0

average_q_values_per_epoch = []

state_list = []
cart_positions = []
pole_angles = []
actions = []

#Videp setup
#video_filename = "dqn_cartpole.avi"
#frame_width, frame_hight = SCREEN_WIDTH, SCREEN_WIDTH
#fps = 30
#fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#out = cv2.VideoWriter(video_filename, fourcc, fps, (frame_width, frame_hight))

for i in range(EPOCHS):
    # Initialize environment 
    world.reset()

    # Get current state
    last_screen = environment.get_screen(world, SCREEN_WIDTH, DEVICE)
    current_screen = environment.get_screen(world, SCREEN_WIDTH, DEVICE)
 
    last_screen = normalize(last_screen)
    current_screen = normalize(current_screen)
 
    state = current_screen - last_screen
    state_list.append(state)
    #print(state_list)
    total_reward = 0 
    q_values = []
 
    for t in count():
        ## Uncomment to record video
        #frame = world.render(mode='rgb_array')
        #if frame is not None:
            #frame = cv2.resize(frame, (frame_width, frame_hight))
            #frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            #out.write(frame)
            
        # Select and perform an action
        action = agent.select_action(state)
        observation, reward, done, _ = world.step(action.item())
        reward = torch.tensor([reward], device=DEVICE)
        
        cart_position = observation[0]
        pole_angle = observation[2]
        cart_positions.append(cart_position)
        #print(cart_position)
        pole_angles.append(pole_angle)
        actions.append(action.item())
        
        q_value = agent.policy_net(state).max(1)[0].item()
        q_values.append(q_value)
        #print(q_value)

        total_reward += reward.item()
        #print(total_reward)

        # Observe new state
        last_screen = current_screen
        current_screen = environment.get_screen(world, SCREEN_WIDTH, DEVICE)

        if not done:
            next_state = normalize(current_screen) - normalize(last_screen)
        else:
            next_state = None
        
        # Store the transition in memory
        agent.memory.push(state, action, next_state, reward)
        
        # Move to the next state
        state = next_state
        
        # Optimize the target network
        agent.optimize_model()
        
        world.render()
        time.sleep(0.01)
        
        if done:
            durations.append(t + 1)
            episode_rewards.append(total_reward)
            break
        
    if len(agent.memory) > 0:
        avg_q_value = sum(q_values) / len(q_values)
    else:
        avg_q_value = 0
    average_q_values_per_epoch.append(avg_q_value)
    
    if i % TARGET_UPDATE == 0:
        agent.target_net.load_state_dict(agent.policy_net.state_dict())
        

#out.release()
world.render()
world.close()

print(durations)

import matplotlib.pyplot as plt

## To obtain the plots in the report have the following settings in the settiings.py file:
#DEVICE = torch.device('cpu')
#SCREEN_WIDTH = 600
#TARGET_UPDATE = 10
#EPOCHS = 100
#BATCH_SIZE = 32
#GAMMA = 0.92
#EPSILON_START = 0.9
#EPSILON_END = 0.05
#EPSILON_DECAY = 50

fig, axs = plt.subplots(2,2, figsize=(7,12))


# Plot 1: Episode Duration
#plt.figure(figsize=(10,6))
axs[0,0].plot(range(len(episode_rewards)), episode_rewards)

axs[0,0].set_title('Learning Curve')
axs[0,0].set_xlabel('Episode')
axs[0,0].set_ylabel('Duration')
axs[0,0].legend()


# Average Rewards 
window_size = 10
rolling_avg_rewards = [sum(episode_rewards[i:i+window_size])/window_size for i in range(0, len(episode_rewards)-window_size +1)]

axs[0,1].plot(range(window_size-1, len(episode_rewards)), rolling_avg_rewards, label = "Rolling Average Rewards (window = 10)", linestyle= '--')
axs[0,1].set_title('Rolling Average Rewards')
axs[0,1].set_xlabel('Episode')
axs[0,1].set_ylabel('Average Rewards')
#axs[0,1].legend()


# Plot Q-values
axs[1,0].plot(range(EPOCHS), average_q_values_per_epoch, label='Q-Values')
axs[1,0].set_title('Average Q-Values Across Epochs')
axs[1,0].set_xlabel('Epochs')
axs[1,0].set_ylabel('Average Q-Value')
#axs[1,0].legend()

# Loss per epidose
#plt.figure(figsize=(10,6))
#print(agent.losses)
axs[1,1].plot(agent.losses, label='Loss')
axs[1,1].set_title('Training Loss')
axs[1,1].set_xlabel('Training Step')
axs[1,1].set_ylabel('Loss')
#plt.legend()
#plt.show() 


# Calculate average loss per episode
#episode_losses = []
#current_step = 0 

#for i in range(len(durations)):
    #steps_in_episode = durations[i]
    #if steps_in_episode > 0:
        #episode_loss = sum(agent.losses[current_step:current_step + steps_in_episode])/steps_in_episode
        #episode_losses.append(episode_loss)
    #current_step += steps_in_episode


#axs[1,1].plot(range(len(episode_losses)), episode_losses)
#axs[1,1].set_title('Average Loss Per Episode')
#axs[1,1].set_xlabel('Episode')
#axs[1,1].set_label('Average Loss')
#axs[1,1].legend()

plt.tight_layout()
plt.show()


# Episode Duration MA
plt.figure(figsize=(10,6))
plt.plot(durations, label='Episode Duration')

window_size = 5
moving_avg = [sum(durations[i:i+window_size])/window_size for i in range(0, len(durations)-window_size +1)]
plt.plot(range(window_size-1, len(durations)), moving_avg, label = "Moving Average (window = 10)")

plt.title('Learning Curve')
plt.xlabel('Episode')
plt.ylabel('Duration')
plt.legend()
plt.show()

plt.figure(figsize=(10,6))
plt.plot(cart_positions, label='Cart Position')
#plt.show()
#plt.figure(figsize=(10,6))
plt.plot(pole_angles, label='Pole Angle')
plt.xlabel('Time Step')
plt.ylabel('Observation')
plt.legend()
plt.show()

# State Visits Heatmap. 

import numpy as np
import seaborn as sns
cart_position_bins = np.linspace(-2.4, 2.4, 10)
pole_angle_bins = np.linspace(-0.2, 0.2, 10)

state_visits_counts = np.zeros((len(cart_position_bins)-1, len(pole_angle_bins)-1))


for cart_position, pole_angle in zip(cart_positions, pole_angles):
    cart_position = np.clip(cart_position, cart_position_bins[0], cart_position_bins[-1] - 1e-5)
    pole_angle = np.clip(pole_angle, pole_angle_bins[0], pole_angle_bins[-1] - 1e-5)
    
    cart_indx = np.digitize(cart_position, cart_position_bins) - 1
    pole_indx = np.digitize(pole_angle, pole_angle_bins) - 1

    if 0 <= cart_indx < state_visits_counts.shape[0] and 0 <= pole_indx < state_visits_counts.shape[1]:
        state_visits_counts[cart_indx, pole_indx] += 1
        #print(f"Updated state_visit_counts[{cart_indx}, {pole_indx}] = {state_visits_counts[cart_indx, pole_indx]}")
    #else:
        #print(f"Indices out of range: Cart Index: {cart_indx}, Pole Index: {pole_indx}")

xticklabels = np.round((pole_angle_bins[1:] + pole_angle_bins[:-1])/2, 2)
yticklabels = np.round((cart_position_bins[1:] + cart_position_bins[:-1])/2, 2)

plt.figure(figsize=(10,8))
sns.heatmap(state_visits_counts, 
            annot=False, cmap="YlGnBu",
            xticklabels=xticklabels,
            yticklabels=yticklabels)

plt.title('State Visit Heatmap')
plt.xlabel('Pole Angle')
plt.ylabel('Cart Position')
plt.show()

#print(actions)
# Action Preference Heatmap 
cart_position_bins = np.linspace(-2.4, 2.4, 10)
pole_angle_bins = np.linspace(-0.2, 0.2, 10)

action_counts_left = np.zeros((len(cart_position_bins)-1, len(pole_angle_bins)-1))
action_counts_right = np.zeros((len(cart_position_bins)-1, len(pole_angle_bins)-1))

for cart_position, pole_angle, action in zip(cart_positions, pole_angles, actions):
    cart_position = np.clip(cart_position, cart_position_bins[0], cart_position_bins[-1] - 1e-5)
    pole_angle = np.clip(pole_angle, pole_angle_bins[0], pole_angle_bins[-1] - 1e-5)
    
    cart_indx = np.digitize(cart_position, cart_position_bins) - 1
    pole_indx = np.digitize(pole_angle, pole_angle_bins) - 1

    if 0 <= cart_indx < action_counts_left.shape[0] and 0 <= pole_indx < action_counts_left.shape[1]:
        if action == 0:
            action_counts_left[cart_indx, pole_indx] += 1
        else:
            action_counts_right[cart_indx, pole_indx] += 1

total_counts = action_counts_left + action_counts_right
action_preference = np.divide(action_counts_left, total_counts, where=total_counts != 0)

xticklabels = np.round((pole_angle_bins[1:] + pole_angle_bins[:-1])/2, 2)
yticklabels = np.round((cart_position_bins[1:] + cart_position_bins[:-1])/2, 2)

plt.figure(figsize=(10,8))
sns.heatmap(action_preference, 
            annot=False, cmap="YlGnBu",
            xticklabels=xticklabels,
            yticklabels=yticklabels)

plt.title('Action Preference Heatmap (Moving Left)')
plt.xlabel('Pole Angle')
plt.ylabel('Cart Position')
plt.show()
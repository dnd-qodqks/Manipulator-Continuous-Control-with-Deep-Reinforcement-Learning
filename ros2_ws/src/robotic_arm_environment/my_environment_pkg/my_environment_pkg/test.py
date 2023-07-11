import matplotlib.pyplot as plt
import numpy as np

save_dir = "/home/dndqodqks/ros2_ws/src/robotic_arm_environment/my_environment_pkg"

avg_rewards = np.loadtxt(save_dir+"/data/reward.txt", delimiter=' ')
avg_actor_loss = np.loadtxt(save_dir+"/data/actor_loss.txt", delimiter=' ')
avg_critic_loss = np.loadtxt(save_dir+"/data/critic_loss.txt", delimiter=' ')

plt.figure(figsize=(15, 10))
    
plt.subplot(2, 2, 1)
plt.grid(True)
plt.plot(range(len(avg_rewards)), avg_rewards, label='avg_reward')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.legend()

plt.subplot(2, 2, 2)
plt.grid(True)
plt.plot(range(len(avg_actor_loss)), avg_actor_loss, label='avg_actor_loss')
plt.xlabel('Episode')
plt.ylabel('Actor Loss')
plt.legend()

plt.subplot(2, 2, 3)
plt.grid(True)
plt.plot(range(len(avg_critic_loss)), avg_critic_loss, label='avg_critic_loss')
plt.xlabel('Episode')
plt.ylabel('Critic Loss')
plt.legend()


plt.show()

# import torch
# import numpy as np

# b = torch.tensor([True, 0.00000001, 1.45], dtype=torch.float32).to('cuda')
# print('output tensor:',b)

# b[0] = -1
# print('It can change np.array:',b)

# print(float(True))
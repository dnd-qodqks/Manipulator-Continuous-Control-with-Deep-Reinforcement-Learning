import matplotlib.pyplot as plt
import time
import datetime

total_start_time = time.time()
total_end_time = time.time()
total_sec = total_end_time - total_start_time
total_time = str(datetime.timedelta(seconds=total_sec)).split(".")[0]
print("Total num of episode completed, Exiting ....")
print("Total Time:", total_time)

now = datetime.datetime.now()

episode = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
rewards = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
avg_rewards = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
plt.plot(episode, rewards, label='reward')
plt.plot(episode, avg_rewards, label='avg_reward')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.legend()
plt.savefig(f'../image/reward_{now.year}_{now.month}_{now.day}_{now.hour}_{now.minute}_{now.second}.png', dpi=200, facecolor='#eeeeee', bbox_inches='tight')

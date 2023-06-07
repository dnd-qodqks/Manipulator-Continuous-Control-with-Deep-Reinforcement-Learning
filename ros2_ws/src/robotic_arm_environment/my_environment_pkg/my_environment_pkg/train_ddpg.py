from .DDPG_manipulator import Agent
from .main_rl_environment import MyRLEnvironmentNode

import matplotlib.pyplot as plt
import numpy as np
import time
import datetime

import rclpy

def main(args=None):

    rclpy.init(args=args)
    env = MyRLEnvironmentNode()
    rclpy.spin_once(env)
    
    agent = Agent(is_training=True)
    
    EPISODE = 10
    EPISODE_STEP = 100
    batch_size = 16
    save_dir = "/home/dndqodqks/ros2_ws/src/robotic_arm_environment/my_environment_pkg/image"
    
    rewards = []
    avg_rewards = []
    
    total_start_time = time.time()
    
    for episode in range(EPISODE):
        episode_start_time = time.time()
        
        env.reset_environment_request()
        agent.noise.reset()
        state, _ = env.get_space()
        episode_reward = 0
        time.sleep(1.0)
        
        for step in range(EPISODE_STEP):
            print (f'----------------Episode:{episode+1} Step:{step+1}--------------------')
            
            action = agent.get_action(state)
            env.action_step_service(action)
            new_state, reward  = env.get_space()

            agent.memory.push(state, action, reward, new_state)
            agent.step_training(batch_size)
            
            state = new_state
            episode_reward += reward
            
            if new_state[0] == True: 
				# if done is TRUE means the end-effector reach to goal and environmet will reset
                print ("'----------------Goal Reach'----------------")
                print(f"episode: {episode+1}, step: {step+1} reward: {np.round(episode_reward, decimals=2)}, average _reward: {np.mean(rewards[-10:])}")
                break

            time.sleep(0.5)
            
        rewards.append(episode_reward)
        avg_rewards.append(np.mean(rewards[-10:]))
        
        episode_end_time = time.time()
        episode_sec = episode_end_time - episode_start_time
        episode_time = str(datetime.timedelta(seconds=episode_sec)).split(".")[0]
                
        print(f'Episode {episode+1} Ended')
        print('Episode Time: ', episode_time)
        
    
    total_end_time = time.time()
    total_sec = total_end_time - total_start_time
    total_time = str(datetime.timedelta(seconds=total_sec)).split(".")[0]
    print("Total num of episode completed, Exiting ....")
    print("Total Time:", total_time)
    
    now = datetime.datetime.now()
    
    ep = [i for i in range(EPISODE)]
    plt.plot(ep, rewards, label='reward')
    plt.plot(ep, avg_rewards, label='avg_reward')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    plt.savefig(f'{save_dir}/reward_{now.year}_{now.month}_{now.day}_{now.hour}_{now.minute}_{now.second}.png', dpi=200, facecolor='#eeeeee', bbox_inches='tight')
    
    rclpy.shutdown()

if __name__ == '__main__':
    main()
    
    print("Train Completed!!!")
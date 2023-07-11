from .DDPG_manipulator import Agent
from .main_rl_environment import MyRLEnvironmentNode

import matplotlib.pyplot as plt
import numpy as np
import time
import datetime
import random
import rclpy

def main(args=None):

    rclpy.init(args=args)
    env = MyRLEnvironmentNode()
    rclpy.spin_once(env)
    
    
    try:
        EPISODE = 25
        EPISODE_STEP = 1000
        batch_size = 256
        save_dir = "/home/dndqodqks/ros2_ws/src/robotic_arm_environment/my_environment_pkg"
        
        agent = Agent(ou_noise=True,
                    critic_learning_rate=0.00001,
                    actor_learning_rate=0.00001, 
                    max_memory_size=1000000,
                    hidden_size=512,
                    device='cuda') # 4:26, 2:56
        
        agent.load_weights(save_dir+"/model/actor_755_h_512_best.pt", save_dir+"/model/critic_755_h_512_best.pt")

        rewards = []
        actor_loss = []
        critic_loss = []
        
        avg_rewards = []
        avg_actor_loss = []
        avg_critic_loss = []
        goal_time_step = []
        
        total_start_time = time.time()
        
        for episode in range(EPISODE):
            episode_start_time = time.time()
            
            env.reset_environment_request()
            agent.noise.reset()
            
            state, _ = env.get_space()
            episode_reward = 0
            episode_actor_loss = np.array([])
            episode_critic_loss = np.array([])
            
            for step in range(EPISODE_STEP):
                print (f'----------------Episode:{episode+1} Step:{step+1}--------------------')
                
                if state[1] < -1.570795 or state[1] > 1.570795:
                    state[1] = random.uniform( -1.570795, 1.570795)
                
                action = agent.get_action(state)
                env.action_step_service(action_values=action, second=0.0001)
                new_state, reward  = env.get_space(step)
                agent.memory.push(state, action, reward, new_state)
                actor_loss, critic_loss = agent.step_training(batch_size)
                
                state = new_state
                episode_reward += reward
                
                if actor_loss is not None and critic_loss is not None:
                    episode_actor_loss += actor_loss
                    episode_critic_loss += critic_loss
                
                goal_time_step.append(0)
                
                if actor_loss is not None and critic_loss is not None:
                    print(f'Episode/step: {episode+1}/{step+1}\treward: {reward:.6f}\tactor_loss: {actor_loss:.6f}\tcritic_loss: {critic_loss:.6f}')
                else:
                    print(f'Episode/step: {episode+1}/{step+1}\treward: {reward:.6f}\tactor_loss: {actor_loss}\tcritic_loss: {critic_loss}')
                
                if new_state[0] == True: 
                    # if done is TRUE means the end-effector reach to goal and environmet will reset
                    print ("----------------Goal Reach----------------")
                    print(f"episode: {episode+1}, step: {step+1} reward: {np.round(episode_reward, decimals=2)}, average _reward: {np.mean(rewards[-10:])}")
                    goal_time_step.append(step+1)
                    agent.save_model(save_dir+"/ckp", f"{episode}_{step}")
                    break
                
                # now_time = time.time()
                # if now_time - episode_start_time > 20:
                #     print("----------------Time Out----------------")
                #     break
            
            rewards.append(episode_reward)
            avg_rewards.append(np.mean(rewards[-10:]))
            
            if actor_loss is not None and critic_loss is not None:
                actor_loss = np.append(actor_loss, episode_actor_loss)
                critic_loss = np.append(critic_loss, episode_critic_loss)
                
                avg_actor_loss = np.append(avg_actor_loss, np.mean(actor_loss[-10:]))
                avg_critic_loss = np.append(avg_critic_loss, np.mean(critic_loss[-10:]))
            
            episode_end_time = time.time()
            episode_sec = episode_end_time - episode_start_time
            episode_time = str(datetime.timedelta(seconds=episode_sec)).split(".")[0]
            
            print(f'Episode {episode+1} Ended')
            print('Episode Time: ', episode_time)
            
        agent.save_model(save_dir+"/model")
        
        total_end_time = time.time()
        total_sec = total_end_time - total_start_time
        total_time = str(datetime.timedelta(seconds=total_sec)).split(".")[0]
        print("Total num of episode completed, Exiting ....")
        print("Total Time:", total_time)
        
        now = datetime.datetime.now()
        
        ep = [i for i in range(EPISODE)]
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 2, 1)
        plt.grid(True)
        plt.plot(ep, avg_rewards, label='avg_reward')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.legend()
        
        reward_txt_ndarray = np.loadtxt(save_dir+"/data/reward.txt", delimiter=' ')
        avg_rewards = np.concatenate([reward_txt_ndarray, avg_rewards], 0)
        np.savetxt(save_dir+"/data/reward.txt", avg_rewards, fmt="%.6e")
        
        if actor_loss is not None and critic_loss is not None:
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
            
            actor_loss_txt_ndarray = np.loadtxt(save_dir+"/data/actor_loss.txt", delimiter=' ')
            avg_actor_loss = np.concatenate([actor_loss_txt_ndarray, avg_actor_loss], 0)
            np.savetxt(save_dir+"/data/actor_loss.txt", avg_actor_loss, fmt="%.6e")
            
            critic_loss_txt_ndarray = np.loadtxt(save_dir+"/data/critic_loss.txt", delimiter=' ')
            avg_critic_loss = np.concatenate([critic_loss_txt_ndarray, avg_critic_loss], 0)
            np.savetxt(save_dir+"/data/critic_loss.txt", avg_critic_loss, fmt="%.6e")

        plt.subplot(2, 2, 4)
        plt.grid(True)
        plt.stem(range(len(goal_time_step)), goal_time_step, label='goal')
        plt.xlabel('Time Step')
        plt.ylabel('Goal')
        plt.legend()
        
        plt.savefig(f'{save_dir+"/image"}/ddpg_{now.year}_{now.month}_{now.day}_{now.hour}_{now.minute}_{now.second}.png', dpi=300, facecolor='#eeeeee', bbox_inches='tight')
    
    except:
        agent.save_model(save_dir+"/model")
        print("try-except save model")
       
    rclpy.shutdown()

if __name__ == '__main__':
    main()
    
    print("Train Completed!!!")
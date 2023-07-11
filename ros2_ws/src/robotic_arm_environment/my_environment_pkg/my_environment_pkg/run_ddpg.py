from .DDPG_manipulator import Agent
from .main_rl_environment import MyRLEnvironmentNode

import rclpy
import time

def main(args=None):
    rclpy.init(args=args)
    env = MyRLEnvironmentNode()
    rclpy.spin_once(env)
    
    save_model_dir = "/home/dndqodqks/ros2_ws/src/robotic_arm_environment/my_environment_pkg/model"
    
    agent = Agent(ou_noise=True, 
                  critic_learning_rate=0.001, 
                  actor_learning_rate=0.001,
                  max_memory_size=1000000, 
                  hidden_size=512, 
                  device='cuda')
    
    agent.load_weights(save_model_dir+"/actor_730_h_512.pt", save_model_dir+"/critic_730_h_512.pt")
    
    EPISODE = 1
    EPISODE_STEP = 1000
    
    for episode in range(EPISODE):
        env.reset_environment_request()
        agent.noise.reset()
        
        # env.action_step_service(action_values=[0.7580684423446655, 0.2184891253709793, 0.7792418599128723, 0.9999995827674866, 0.5650008916854858, 1.0], second=0.5)    
        
        state, _ = env.get_space()
        
        for step in range(EPISODE_STEP):
            if state[0] == True: 
				# if done is TRUE means the end-effector reach to goal and environmet will reset
                print ("'----------------Goal Reach'----------------")
                print(f"episode: {episode+1}, step: {step+1}")
                break
            
            print (f'----------------Episode:{episode+1} Step:{step+1}--------------------')

            print("run", state)
            action = agent.get_action(state)
            env.action_step_service(action_values=action, second=0.1)
            state, _  = env.get_space(5)
    
    # save_model_dir = "/home/dndqodqks/ros2_ws/src/robotic_arm_environment/my_environment_pkg/ckp_600"
    
    # file_list = os.listdir(save_model_dir)
    # print(file_list)
    # print(len(file_list))
    
    # min = 100
    # min_model = ""
    # for i, file_name in enumerate(file_list):
    #     if i < len(file_list):
    #         temp = file_name.split('_')
            
    #         agent.load_weights(save_model_dir+f"/actor_{temp[1]}_{temp[2]}", save_model_dir+f"/critic_{temp[1]}_{temp[2]}")
            
    #         env.reset_environment_request()
    #         state, _ = env.get_space()
            
    #         action = agent.get_action(state)
    #         env.action_step_service(action_values=action, second=0.0001)
    #         new_state, reward  = env.get_space(4)
            
    #         distance = env.get_distance()
            
    #         if distance < min:
    #             min = distance
    #             min_model = f"actor_{temp[1]}_{temp[2]}, critic_{temp[1]}_{temp[2]}, {distance}"
    #             print(f"actor_{temp[1]}_{temp[2]}, critic_{temp[1]}_{temp[2]}")
    
    # print(min_model)
    
    rclpy.shutdown()

if __name__ == '__main__':    
    main()
    
    print("Run Completed!!!")
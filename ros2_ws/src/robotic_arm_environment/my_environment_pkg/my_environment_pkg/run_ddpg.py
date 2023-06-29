from .DDPG_manipulator import Agent
from .main_rl_environment import MyRLEnvironmentNode

import rclpy

def main(args=None):
    rclpy.init(args=args)
    env = MyRLEnvironmentNode()
    rclpy.spin_once(env)
    
    save_model_dir = "/home/dndqodqks/ros2_ws/src/robotic_arm_environment/my_environment_pkg/model"
    
    agent = Agent(ou_noise=True, 
                  critic_learning_rate=0.001, 
                  actor_learning_rate=0.001, 
                  tau=0.001, 
                  max_memory_size=1000000, 
                  hidden_size=512, 
                  device='cuda')
    
    agent.load_weights(save_model_dir+"/actor_200_h_512_best.pt", save_model_dir+"/critic_200_h_512_best.pt")
    
    EPISODE = 1
    EPISODE_STEP = 1000
    
    for episode in range(EPISODE):
        env.reset_environment_request()
        agent.noise.reset()
        
        state, _ = env.get_space()
        
        for step in range(EPISODE_STEP):
            print (f'----------------Episode:{episode+1} Step:{step+1}--------------------')

            action = agent.get_action(state)
            env.action_step_service(action_values=action, second=0.0001)
            new_state, reward  = env.get_space(step)
            agent.memory.push(state, action, reward, new_state)
            
            state = new_state
            
            if new_state[0] == True: 
				# if done is TRUE means the end-effector reach to goal and environmet will reset
                print ("'----------------Goal Reach'----------------")
                print(f"episode: {episode+1}, step: {step+1}")
                break
    
    rclpy.shutdown()

if __name__ == '__main__':    
    main()
    
    print("Run Completed!!!")
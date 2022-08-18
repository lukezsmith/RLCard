''' An example of training a reinforcement learning agent on the environments in RLCard
'''
import os
import argparse

import torch

import rlcard
from rlcard.agents import RandomAgent
from rlcard.utils import get_device, set_seed, tournament, reorganize, Logger, plot_curve


def train(env, algorithm, cuda, seed, num_episodes, num_eval_games, evaluate_every, log_dir):

    # hyperparameters I need to parameterise
    # evaluate_every = 100
    # save_plot_every = 1000
    # evaluate_num = 10000
    num_episodes = 10000000
    evaluate_every = 100
    save_plot_every = 1000
    evaluate_num = 10000
    log_path = './logs/log.txt'
    csv_path = './logs/performance.csv'
    figure_path = "./figures/"
    memory_init_size = 1000
    norm_step = 100

    # Check whether gpu is available
    device = get_device()
        
    # Seed numpy, torch, random
    set_seed(seed)

    # Make the training environment with seed
    env = rlcard.make(env, config={'seed': seed, 'allow_step_back': True})
    
    # Make the evaluation environment with seed
    eval_env = rlcard.make("teamuno", config={'seed': seed})

    # Initialize the agent and use random agents as opponents
    from rlcard.agents import CFRAgent
    agents = []

    # create NFSP agents
    for i in range(env.num_players):
      agent = CFRAgent(env)
      agents.append(agent)
    
    # set agents for training
    env.set_agents(agents)

    # for _ in range(1, env.num_players):
    #     agents.append(RandomAgent(num_actions=env.num_actions))
    # env.set_agents(agents)

    # create random agents for evaluation
    random_agent_1 = RandomAgent(num_actions=env.num_actions)
    random_agent_2 = RandomAgent(num_actions=env.num_actions)

    eval_env.set_agents([agents[0], random_agent_1, agents[2], random_agent_2])

    # variable to count number of timesteps
    # step_counters = [0 for _ in range(env.num_players)] 

    # Start training
    # logger = Logger(xlabel='timestep', ylabel='reward', legend='NFSP on Leduc Holdem', log_path=log_path, csv_path=csv_path)
    # logger = Logger(log_path)

    with Logger(log_dir) as logger:
      for episode in range(num_episodes):

          # sample policy for episode for each 
          # for agent in agents:
          #   agent.sample_episode_policy()

          # # Generate data from the environment
          # trajectories, payoffs, _ = env.run(is_training=True)

          # # Reorganaize the data to be state, action, reward, next_state, done
          # trajectories = reorganize(trajectories, payoffs)

          # Feed transitions into agent memory, and train the agent
          # Here, we assume that DQN always plays the first position
          # and the other players play randomly (if any)
          for i in range(env.num_players):
            agents[i].train()
            # for ts in trajectories[i]:
                # agents[i].feed(ts)
                # step_counters[i] +=1

                # train agent
                # train_count = step_counters[i] - (memory_init_size + norm_step)

                # TODO: learning rates are actually defined in the NFSPAgent class, 
                # need to set this when we instantiate nfsp agent
                # train supervised agent
                # if train_count > 0 and train_count % 64 == 0:
                #   # print(agents[i])
                #   # rl_loss = agents[i].train_rl()
                #   sl_loss = agents[i].train_sl()
                #   print('\rINFO - Agent {}, step {}, sl-loss: {}'.format(i, step_counters[i], sl_loss), end='')

          # Evaluate the performance. Play with random agents.
          if episode % evaluate_every == 0:
              payoffs, winnings = tournament(env, num_eval_games)
              # logger.log_performance(env.timestep, payoffs[0] + payoffs[2])
              # print(winnings)
              logger.log_winrate(env.timestep, payoffs[0] + payoffs[2], winnings[0])
              # logger.log_performance(env.timestep, tournament(env, num_eval_games)[0])
              # reward = 0
              # for eval_episode in range(evaluate_num):
              #     _, payoffs = eval_env.run(is_training=False)
              #     reward += payoffs[0]

              # logger.log('\n########## Evaluation ##########')
              # logger.log('Timestep: {} Average reward is {}'.format(env.timestep, float(reward)/evaluate_num))

              # Add point to logger
              # logger.add_point(x=env.timestep, y=float(reward)/evaluate_num)

          # Make plot
          if episode % save_plot_every == 0 and episode > 0:
              # logger.make_plot(save_path=figure_path+str(episode)+'.png')                
              # logger.plot('nfps')
              plot_curve(csv_path, figure_path, "nfps")


              

      # Get the paths
      # csv_path, fig_path = logger.csv_path, logger.fig_path

    # Plot the learning curve
    # plot_curve(csv_path, fig_path, algorithm)

    # Save model
    save_path = os.path.join(log_dir, 'model.pth')
    torch.save(agent, save_path)
    print('Model saved in', save_path)

if __name__ == '__main__':
    env = "teamuno"
    algorithm = "nfsp"
    cuda = ''
    seed = 42
    num_episodes = 500
    num_eval_games = 50
    evaluate_every = 100
    log_dir = './logs/'

    # device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


    train( env, algorithm, cuda, seed, num_episodes, num_eval_games, evaluate_every, log_dir)
    
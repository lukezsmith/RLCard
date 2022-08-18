import os
import argparse

import torch

import rlcard
from rlcard.agents import RandomAgent
from rlcard.models.leducholdem_rule_models import LeducHoldemRuleAgentV2

from rlcard.utils import get_device, set_seed, tournament, tournament_winnings, reorganize, Logger, plot_winrate_curve
import time

def train(env, algorithm, device, seed, num_episodes, num_eval_games, evaluate_every, save_every, log_dir, opposition_agent):

    # hyperparameters I need to parameterise
    csv_path = log_dir + opposition_agent + '_performance.csv'
#     print(csv_path)
#     figure_path = log_dir + opposition_agent + '_fig.png'
#     print(figure_path)
#     figure_path = './figures/'

    # Seed numpy, torch, random
    set_seed(seed)

    # Make the training environment with seed
    env = rlcard.make(env, config={'seed': seed})

    # Make the evaluation environment with seed
    eval_env = rlcard.make("teamunounshaped", config={'seed': seed})

    # Initialize the agents
    if algorithm == "NFSP":
        from rlcard.agents import NFSPAgent
        agents = []

        # create NFSP agents
        for i in range(env.num_players):
            agent = NFSPAgent(num_actions=env.num_actions,
                              state_shape=env.state_shape[0],
                              hidden_layers_sizes=[64, 64],
                              q_mlp_layers=[64, 64],
                              device=device)
            agents.append(agent)
    elif algorithm == "NFSP-ARM":
        from rlcard.agents import NFSPARMAgent
        agents = []

        # create NFSP agents
        for i in range(env.num_players):
            agent = NFSPARMAgent(num_actions=env.num_actions,
                                 state_shape=env.state_shape[0],
                                 hidden_layers_sizes=[64, 64],
                                 q_mlp_layers=[64, 64],
                                 device=device)
            agents.append(agent)

    elif algorithm == "DQN":
        from rlcard.agents import DQNAgent
        agents = []

        # create NFSP agents
        for i in range(env.num_players):
            agent = DQNAgent(num_actions=env.num_actions,
                             state_shape=env.state_shape[0],
                             mlp_layers=[64, 64],
                             device=device)
            agents.append(agent)
    else:
        agents = []

        # create NFSP agents
        for i in range(env.num_players):
            agent = DQNAgent(num_actions=env.num_actions,
                             state_shape=env.state_shape[0],
                             mlp_layers=[64, 64],
                             device=device)
            agents.append(agent)

    # set agents for training
    env.set_agents(agents)

    # for _ in range(1, env.num_players):
    #     agents.append(RandomAgent(num_actions=env.num_actions))
    # env.set_agents(agents)

    if opposition_agent == "rule-based":
        # create random agents for evaluation
        eval_agent_1 = LeducHoldemRuleAgentV2()
        eval_agent_2 = LeducHoldemRuleAgentV2()

    else:
        # create random agents for evaluation
        eval_agent_1 = RandomAgent(num_actions=env.num_actions)
        eval_agent_2 = RandomAgent(num_actions=env.num_actions)

    eval_env.set_agents([agents[0], eval_agent_1, agents[2], eval_agent_2])

    # Start training
    with Logger(log_dir ,opposition_agent) as logger:
        for episode in range(num_episodes):

            # sample policy for episode for each
#             for agent in agents:
#                 agent.sample_episode_policy()

            # Generate data from the environment
            trajectories, payoffs, _ = env.run(is_training=True)

            # Reorganaize the data to be state, action, reward, next_state, done
            trajectories = reorganize(trajectories, payoffs)

            # Feed transitions into agent memory, and train the agent
            # Here, we assume that DQN always plays the first position
            # and the other players play randomly (if any)
            for i in range(env.num_players):
                for ts in trajectories[i]:
                    agents[i].feed(ts)

            # Evaluate the performance. Play with random agents.
            if episode % evaluate_every == 0:
                payoffs, winnings = tournament_winnings(eval_env, num_eval_games)
                # logger.log_performance(env.timestep, payoffs[0] + payoffs[2])
                logger.log_winrate(
                    env.timestep, payoffs[0] + payoffs[2], winnings[0])

            # Make plot
            if episode % save_every == 0 and episode > 0:
#                 plot_curve(csv_path, figure_path, algorithm)

                # Save model
                save_path = os.path.join(log_dir, opposition_agent +'_model.pth')
                torch.save(agents[0], save_path)
            csv_path, fig_path = logger.csv_path, logger.fig_path
    plot_winrate_curve(csv_path, fig_path, algorithm)


if __name__ == '__main__':
    env = "teamunounshaped"
    algorithm = "DQN"

    # hyperparameters
    device = get_device()
    seed = 42
    # num_episodes = 1000
    # num_eval_games = 250
    num_episodes = 250
    num_eval_games = 62
    evaluate_every = 50
    save_every = 50
    opposition_agent = 'random'
    log_dir = './'

    start = time.time()
    train(env, algorithm, device, seed, num_episodes,
        num_eval_games, evaluate_every, save_every, log_dir, opposition_agent)
    end = time.time()
    print(end - start)

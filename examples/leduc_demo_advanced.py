"""
This code is largely a modified version of the original RLCard github repo: https://github.com/datamllab/rlcard.
We also use and modify the ARM implementation provided in the accompanying code: https://github.com/kxinhe/LRM_FP of the following paper: 
"He, H. Wu, Z. Wang, and H. Li, “Finding nash equilibrium
for imperfect information games via fictitious play based on local regret minimization,” 
International Journal of Intelligent Systems, 2022"
in our NFSP-ARM implementation. 
"""
import os
import rlcard
from rlcard import models
from rlcard.utils import print_card, get_device
import numpy as np
import time

if __name__ == '__main__':

    # Make environment
    # Set 'record_action' to True because we need it to print results
    env = rlcard.make('teamleducholdemunshaped', config={'record_action': True})
    device = get_device()

    # load model from checkpoint
    def load_model(model_path, env=None, position=None, device=None):
        if os.path.isfile(model_path):  # Torch model
            import torch
            agent = torch.load(model_path, map_location=device)
            agent.set_device(device)
        elif os.path.isdir(model_path):  # CFR model
            from rlcard.agents import CFRAgent
            agent = CFRAgent(env, model_path)
            agent.load()
        elif model_path == 'random':  # Random model
            from rlcard.agents import RandomAgent
            agent = RandomAgent(num_actions=env.num_actions)
        else:  # A model in the model zoo
            from rlcard import models
            agent = models.load(model_path).agents[position]
        
        return agent

    # load models from saved model
    model_path = './leduc_model_random.pth'
    agents = []
    print("Please wait, loading solution models and initialising environment...")
    agent1 = load_model(model_path, device=device)
    agent2 = load_model(model_path, device=device)
    random_agent1 = models.load('leduc-holdem-cfr').agents[0]
    random_agent2 = models.load('leduc-holdem-cfr').agents[0]


    agents  = [agent1, random_agent1, agent2, random_agent2]
    env.set_agents(agents)

    print(">> Team Leduc Hold'em Demonstration")
    num_wins = 0
    num_games = 0

    while (True):
        print(">> Start a new game")
        num_games +=1
        trajectories, payoffs, _ = env.run(is_training=False)
        # If the human does not take the final action, we need to
        # print other players action
        final_state = trajectories[0][-1]
        action_record = final_state['action_record']
        state = final_state['raw_obs']
        _action_list = []
        for i in range(1, len(action_record)+1):
            _action_list.insert(0, action_record[-i])
        for pair in _action_list:
            if pair[0] % 2 == 0:
                player_name = "Player " + str(pair[0]) + " (NFSP Solution Team)"
            else:
                player_name = "Player " + str(pair[0]) + " (Pre-trained Team)"
            print('>> ' ,player_name, 'chooses', pair[1])
            print('With hand: ')
            print_card(env.get_perfect_information()['hand_cards'][pair[0]])
            
            time.sleep(1.5)

        print('===============     Result     ===============')
        if len(set(payoffs)) == 1:
            print("Game was a draw!")
        else:

            winner = np.argmax(payoffs)
            if winner == 0 or winner == 2:
                winning_team = "NFSP Solution Team"
                num_wins +=1
            else:
                winning_team = "Pre-trained Agent Team"
            
            print(winning_team + " won the game through agent "+ str(winner) + " with hand: ")
            print_card(env.get_perfect_information()['hand_cards'][winner])
        win_rate = num_wins / num_games
        print("NFSP Solution Team Win Rate: ", str(win_rate))
            

        input("Press any key to move to next game...")
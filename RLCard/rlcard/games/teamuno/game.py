from copy import deepcopy
import numpy as np

from rlcard.games.teamuno import Dealer
from rlcard.games.teamuno import Player
from rlcard.games.teamuno import Round


class UnoGame:

    def __init__(self, allow_step_back=False):
        # need to implement a check here for team construction: needs to be teams of 2
        # hence, even number of players
        self.allow_step_back = allow_step_back
        self.np_random = np.random.RandomState()
        self.team_lengths = [2, 2]
        self.num_players = 4
        self.payoffs = [0 for _ in range(self.num_players)]

    def configure(self, game_config):
        ''' Specifiy some game specific parameters, such as number of players
        '''
        self.num_players = game_config['game_num_players']

    def init_game(self):
        ''' Initialize players and state

        Returns:
            (tuple): Tuple containing:

                (dict): The first state in one game
                (int): Current player's id
        '''
        # Initalize payoffs
        self.payoffs = [0 for _ in range(self.num_players)]

        # Initialize a dealer that can deal cards
        self.dealer = Dealer(self.np_random)

        # Initialize four players to play the game
        self.players = [Player(i, self.np_random) for i in range(self.num_players)]

        # Deal 7 cards to each player to prepare for the game
        for player in self.players:
            self.dealer.deal_cards(player, 7)

        # Initialize a Round
        self.round = Round(self.dealer, self.np_random)

        # flip and perfrom top card
        top_card = self.round.flip_top_card()
        self.round.perform_top_card(self.players, top_card)

        # Save the hisory for stepping back to the last state.
        self.history = []

        player_id = self.round.current_player
        state = self.get_state(player_id)
        return state, player_id

    def step(self, action):
        ''' Get the next state

        Args:
            action (str): A specific action

        Returns:
            (tuple): Tuple containing:

                (dict): next player's state
                (int): next plater's id
        '''

        if self.allow_step_back:
            # First snapshot the current state
            his_dealer = deepcopy(self.dealer)
            his_round = deepcopy(self.round)
            his_players = deepcopy(self.players)
            self.history.append((his_dealer, his_players, his_round))

        self.round.proceed_round(self.players, action)
        player_id = self.round.current_player
        state = self.get_state(player_id)
        return state, player_id

    def step_back(self):
        ''' Return to the previous state of the game

        Returns:
            (bool): True if the game steps back successfully
        '''
        if not self.history:
            return False
        self.dealer, self.players, self.round = self.history.pop()
        return True

    def get_state(self, player_id):
        ''' Return player's state

        Args:
            player_id (int): player id

        Returns:
            (dict): The state of the player
        '''
        state = self.round.get_state(self.players, player_id)
        state['num_players'] = self.get_num_players()
        state['current_player'] = self.round.current_player
        return state

    def get_payoffs(self, algorithm="nfsp", partial_rewards=True):
        ''' Return the payoffs of the game

        Returns:
            (list): Each entry corresponds to the payoff of one player
        '''

        winner_reward = 2
        shed_reward = 1
        excess_card_penalty = 0.01

        winner = self.round.winner
        shed_players = self.round.shed_players
        winner_indices = set(winner)
        loser_indices = [i for i in range(len(self.payoffs)) if i not in winner_indices]
        winners = [1  if i in winner else 0 for i in range(self.num_players)  ]

        self.payoffs[winner[0]] = winner_reward
        self.payoffs[winner[1]] = winner_reward

        # variable to track the total excess card penalty so the reverse can be distributed (evenly) to winners
        # so we are zero-sum
        total_excess_penalties = 0

        for i in loser_indices:
            player_card_amount = len(self.players[i].hand)

            # distribute losing penalty
            self.payoffs[i] = -winner_reward
            self.payoffs[i] -= shed_reward

            # give shed player positive reward and losing player negative
            if (player_card_amount) == 0:
                self.payoffs[i] += (2*shed_reward)
                self.payoffs[(i + 2) % self.num_players] -= (2*shed_reward)


            penalty = (excess_card_penalty * player_card_amount)
            total_excess_penalties += penalty

            # distribute excess card penalty
            self.payoffs[i] -= penalty

        # give winners excess card bonus
        self.payoffs[winner[0]] += (total_excess_penalties /2)
        self.payoffs[winner[1]] += (total_excess_penalties /2)

        # distribute partial rewards for shed players
        if partial_rewards:
            for player in self.round.shed_players:
                if player not in loser_indices:
                    self.payoffs[player] += shed_reward

        return self.payoffs, winners

    def get_legal_actions(self):
        ''' Return the legal actions for current player

        Returns:
            (list): A list of legal actions
        '''

        return self.round.get_legal_actions(self.players, self.round.current_player)

    def get_num_players(self):
        ''' Return the number of players in Limit Texas Hold'em

        Returns:
            (int): The number of players in the game
        '''
        return self.num_players

    @staticmethod
    def get_num_actions():
        ''' Return the number of applicable actions

        Returns:
            (int): The number of actions. There are 61 actions
        '''
        return 61

    def get_player_id(self):
        ''' Return the current player's id

        Returns:
            (int): current player's id
        '''
        return self.round.current_player

    def is_over(self):
        ''' Check if the game is over

        Returns:
            (boolean): True if the game is over
        '''
        return self.round.is_over

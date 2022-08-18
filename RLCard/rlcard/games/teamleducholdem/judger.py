from rlcard.utils.utils import rank2int

class LeducholdemJudger:
    ''' The Judger class for Leduc Hold'em
    '''
    def __init__(self, np_random):
        ''' Initialize a judger class
        '''
        self.np_random = np_random

    @staticmethod
    def judge_game( players, public_card):
        ''' Judge the winner of the game.

        Args:
            players (list): The list of players who play the game
            public_card (object): The public card that seen by all the players

        Returns:
            (list): Each entry of the list corresponds to one entry of the
        '''
        winner = None

        # Judge who are the winners
        fold_count = 0
        ranks = []
        # If every player folds except one, the alive player is the winner
        for idx, player in enumerate(players):
            ranks.append(rank2int(player.hand.rank))
            if player.status == 'folded':
               fold_count += 1
            elif player.status == 'alive':
                alive_idx = idx
        if fold_count == (len(players) - 1):
            winner = alive_idx
        
        # If any of the players matches the public card wins
        if winner == None:
            for idx, player in enumerate(players):
                if player.hand.rank == public_card.rank:
                    winner = idx 
                    break
        
        # If non of the above conditions, the winner player is the one with the highest card rank
        if winner == None:
            max_rank = max(ranks)
            max_index = [i for i, j in enumerate(ranks) if j == max_rank]
            for idx in max_index:
                winner = idx
        return winner

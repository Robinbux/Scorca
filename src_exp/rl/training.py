import os
from reconchess import Player, play_local_game
from reconchess.history import GameHistory
from src.rl.rl_bot import RLBot
from src.rl.rl_agent import RLAgent
from src.utils import bool_to_color, logger

SECONDS_PER_PLAYER = 100
BATCH_SIZE = 32
NUM_GAMES = 1


class GameMaster:
    GAME_LOGS_PATH = '../game_logs'

    def play_single_game(self, white_bot: Player, black_bot: Player, game_id: int) -> str:
        logger.info(f"Playing game {game_id}...\n"
                    f"White player: {white_bot}\n"
                    f"Black player: {black_bot}")

        winner_color, win_reason, history = play_local_game(white_player=white_bot, black_player=black_bot,
                                                            seconds_per_player=SECONDS_PER_PLAYER)
        history_file = self.save_game_history(history, game_id)

        logger.info(f'Game {game_id}: {white_bot if winner_color else black_bot} ({bool_to_color(winner_color)}) '
                    f'won because of {win_reason}')
        logger.info(f'Saving history to {history_file}')

        return history_file, winner_color, win_reason

    def save_game_history(self, history: GameHistory, game_id: int) -> str:
        history_path = os.path.join(self.GAME_LOGS_PATH, f'game_{game_id}')
        os.makedirs(history_path, exist_ok=True)

        history_file = os.path.join(history_path, 'history.json')
        history.save(history_file)

        return history_file

    def run_experiment(self, num_games: int):
        rl_agent = RLAgent()
        rl_bot_white = RLBot(rl_agent)
        rl_bot_black = RLBot(rl_agent)

        for game_id in range(num_games):
            logger.info(f'Game {game_id}: RL Bot (white) vs RL Bot (black)')

            history_file, winning_color, win_reason = self.play_single_game(rl_bot_white, rl_bot_black, game_id)

            logger.info(f'Game {game_id} winner: {winning_color}')
            logger.info('Training RL agent...')
            rl_agent.replay(BATCH_SIZE)

        logger.info('Saving RL agent...')
        rl_agent.save('rl_agent.h5')


if __name__ == '__main__':
    print("START")
    game_master = GameMaster()
    game_master.run_experiment(NUM_GAMES)

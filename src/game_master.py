import json
import os
from collections import defaultdict
from enum import Enum

import chess
from reconchess import Player, play_local_game, WinReason
from reconchess.bots.attacker_bot import AttackerBot
from reconchess.bots.trout_bot import TroutBot
from reconchess.history import GameHistory

from src.misc_bots.preset_move_bot import PresetMoveBot
from src.scorca.scorca import Scorca

from loguru import logger


import random

from src.scorca.utils import bool_to_color

SECONDS_PER_PLAYER = 300

class BotOptions(str, Enum):
    SCORCA = 'SCORCA'
    STRANGE_FISH = 'STRANGE_FISH'
    TROUT = 'TROUT'
    ATTACKER = 'ATTACKER'
    PRESET_MOVE = 'PRESET_MOVE'


class GameMaster:
    GAME_LOGS_PATH = '../game_logs'

    @staticmethod
    def setup_bot(bot_option: BotOptions):
        bots = {
            BotOptions.TROUT: lambda: ('Trout Bot', TroutBot()),
            BotOptions.ATTACKER: lambda: ('Attacker Bot', AttackerBot()),
            BotOptions.STRANGE_FISH: lambda: ('Strange Fish', StrangeFish2()),
            BotOptions.SCORCA: lambda: ('SCORCA', Scorca()),
            BotOptions.PRESET_MOVE: lambda: ('Preset Move Bot', PresetMoveBot()),
        }

        bot_name, bot = bots[bot_option]()

        return bot_name, bot

    def play_single_game(self, white_bot: Player, black_bot: Player, game_id: int, experiment_dir: str) -> tuple[
        str, bool | None, WinReason | None, int]:
        logger.info(f"Playing game {game_id}...\n"
                    f"White player: {white_bot}\n"
                    f"Black player: {black_bot}")

        winner_color, win_reason, history = play_local_game(white_player=white_bot, black_player=black_bot,
                                                            seconds_per_player=SECONDS_PER_PLAYER)
        history_file = self.save_game_history(history, game_id, experiment_dir)

        logger.info(f'Game {game_id}: {white_bot if winner_color else black_bot} ({bool_to_color(winner_color)}) '
                    f'won because of {win_reason}')
        logger.info(f'Game lasted {history.num_turns()} turns')
        logger.info(f'Saving history to {history_file}')

        return history_file, winner_color, win_reason, history.num_turns(chess.WHITE)

    def save_game_history(self, history: GameHistory, game_id: int, experiment_dir: str) -> str:
        history_path = os.path.join(experiment_dir, f'game_{game_id}')
        os.makedirs(history_path, exist_ok=True)

        history_file = os.path.join(history_path, 'history.json')
        history.save(history_file)

        return history_file

    def get_game_winner(self, history_file: str):
        with open(history_file, 'r') as file:
            history_data = json.load(file)
            winning_color = history_data['winner_color']
        return winning_color

    def run_experiment(self, num_games: int, bot1_options: BotOptions, bot2_options: BotOptions,
                       bot1_params=None, bot2_params=None):
        # Create new experiment directory
        experiment_dir = self.create_new_experiment_dir()


        results = defaultdict(int)

        # Set up the bots with hardcoded parameters if provided
        bot1_name, bot1 = self.setup_bot(bot1_options)
        bot1_name = f"BOT 1 ({bot1_name})"
        bot2_name, bot2 = self.setup_bot(bot2_options)
        bot2_name = f"BOT 2 ({bot2_name})"

        self.save_initial_experiment_settings(experiment_dir, num_games, bot1, bot2)

        game_details_list = []

        for game_id in range(num_games):
            # Randomly assign bots to colors
            is_bot1_white = random.random() < 0.5

            white, white_name, black, black_name = (bot1, bot1_name, bot2, bot2_name) if is_bot1_white else (bot2, bot2_name, bot1, bot1_name)

            logger.info(f'Game {game_id}: {white} (white) vs {black} (black)')

            history_file, winning_color, win_reason, turns = self.play_single_game(white, black, game_id, experiment_dir)

            winner = white_name if winning_color else black_name
            logger.info(f'Game {game_id} winner: {winner}')
            results[winner] += 1

            # Save details of each game
            game_details = {
                'game_id': game_id,
                'white_bot': white_name,
                'black_bot': black_name,
                'winner': winner,
                'winning_color': bool_to_color(winning_color),
                'win_reason': str(win_reason),
                'turns': turns,
            }

            if hasattr(white, 'out_of_time'):
                game_details['white_went_timeout'] = white.out_of_time
            if hasattr(black, 'out_of_time'):
                game_details['black_went_timeout'] = black.out_of_time

            game_details_list.append(game_details)

            # Update results after each iteration
            self.update_results_and_game_details(experiment_dir, dict(results), game_details_list)

            self.reset_bots(bot1, bot2)

        # Return win rate of bot1
        return results[bot1_name] / num_games


    def reset_bots(self, bot1: Player, bot2: Player):
        if isinstance(bot1, Scorca):
            bot1.reset()
        if isinstance(bot2, Scorca):
            bot2.reset()


    def update_results_and_game_details(self, dir_path, results, game_details):
        settings_file_path = os.path.join(dir_path, 'settings.json')

        # Read the existing data
        with open(settings_file_path, 'r') as file:
            data = json.load(file)

        # Update the results
        data['results'] = results
        data['game_details'] = game_details

        # Write back to the file
        with open(settings_file_path, 'w') as file:
            json.dump(data, file, indent=2)


    def create_new_experiment_dir(self):
        new_dir_index = len(os.listdir(self.GAME_LOGS_PATH))
        new_dir_name = 'games_{:03d}'.format(new_dir_index)
        new_dir_path = os.path.join(self.GAME_LOGS_PATH, new_dir_name)
        os.makedirs(new_dir_path, exist_ok=True)

        # self.GAME_LOGS_PATH = new_dir_path  # Update GAME_LOGS_PATH to point to the new experiment directory

        return new_dir_path

    def save_initial_experiment_settings(self, dir_path, num_games, bot1, bot2):
        settings_file_path = os.path.join(dir_path, 'settings.json')
        with open(settings_file_path, 'w') as file:
            json.dump({
                'num_games': num_games,
                'seconds_per_player': SECONDS_PER_PLAYER,
            }, file, indent=2)


if __name__ == '__main__':
    print("START")
    game_master = GameMaster()

    num_experiments = 1
    num_games = 1


    bot1 = BotOptions.SCORCA
    bot2 = BotOptions.ATTACKER


    for i in range(num_experiments):
        print(f"Running experiment {i + 1}...")

        game_master.run_experiment(
            num_games=num_games,
            bot1_options=bot1,
            bot2_options=bot2,
        )

import json
import os
import random

import chess
from reconchess import play_local_game
from tqdm import tqdm

from scorca.scorca import Scorca

script_dir = os.path.dirname(os.path.realpath(__file__))
EXPERIMENT_SENSE_DIR = os.path.join(script_dir, '..', 'experiments', 'sense_comp', 'history')

# Define a directory where your json files are located

json_dir = os.path.join(script_dir, '..', 'neurips2022_RBC_game_logs')

# Get a list of all json files in the directory
json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]
json_files.sort()

START_IDX = 0
END_IDX = 60
def main():
    for idx, json_file in tqdm(enumerate(json_files[START_IDX:END_IDX], start=START_IDX)):
        # Load the json file
        with open(os.path.join(json_dir, json_file), 'r') as f:
            game_data = json.load(f)



        # If it's too long we skip
        if len(game_data['senses']['true']) > 20 or game_data['win_reason']['value'] != 'KING_CAPTURE':
            continue

        save_white = True
        save_black = True

        if game_data['white_name'] in ['random', 'attacker']:
            save_white = False

        if game_data['black_name'] in ['random', 'attacker']:
            save_black = False




        #white_player = HistoryBot(f'{json_dir}/{json_file}', chess.WHITE, f'{EXPERIMENT_ENTROPY_DIR}/{str(idx).zfill(3)}_white')
        #black_player = HistoryBot(f'{json_dir}/{json_file}', chess.BLACK, f'{EXPERIMENT_ENTROPY_DIR}/{str(idx).zfill(3)}_black')

        white_player = Scorca(history_path=f'{json_dir}/{json_file}', results_path=f'{EXPERIMENT_SENSE_DIR}/{str(idx).zfill(3)}_white', save_results=save_white)
        black_player = Scorca(history_path=f'{json_dir}/{json_file}', results_path=f'{EXPERIMENT_SENSE_DIR}/{str(idx).zfill(3)}_black', save_results=save_black)


        # Play a local game
        winner_color, win_reason, history = play_local_game(white_player=white_player, black_player=black_player)

if __name__=='__main__':
    main()

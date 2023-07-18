import json
import os
from typing import List, Dict
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import chess

NEURIPS_FIR_DIR = '/Users/robinbux/Desktop/RBC_New/new_history_game_logs/'


BOTS_TO_LOG = [
    'JKU-CODA',
    'Châteaux',
    'Fianchetto',
    'trout',
    'StrangeFish2',
    'random'
]

# Data storage for the distance analysis
distance_data = {}

# Iterate through neurips fir and all JSON files
for filename in os.listdir(NEURIPS_FIR_DIR):
    if filename.endswith(".json"):
        # Read in the JSON file
        with open(NEURIPS_FIR_DIR + filename, 'r') as f:
            json_data = json.load(f)
        black_bot_name: str = json_data['black_name']
        white_bot_name: str = json_data['white_name']
        black_senses: List[int] = json_data['senses']['false']
        white_senses: List[int] = json_data['senses']['true']
        black_requested_moves: List[Dict[str, str]] = json_data['requested_moves']['false']
        white_requested_moves: List[Dict[str, str]] = json_data['requested_moves']['true']

        # Initialize data storage for these bots if not present
        if black_bot_name not in distance_data:
            distance_data[black_bot_name] = {"total_distance": {}, "counts": {}}
        if white_bot_name not in distance_data:
            distance_data[white_bot_name] = {"total_distance": {}, "counts": {}}

        fens_during_the_game: List[str] = [x for pair in zip(json_data['fens_before_move']['true'],
                                                             json_data['fens_before_move']['false']) for x in pair]

        if len(json_data['fens_before_move']['true']) > len(json_data['fens_before_move']['false']):
            fens_during_the_game.append(json_data['fens_before_move']['true'][-1])

            # Analysis for all different bots, about the distance moved per piece type on average in games
            for i, fen in enumerate(fens_during_the_game):
                board = chess.Board(fen)

                # Determine which bot is playing
                bot_name = white_bot_name if board.turn else black_bot_name
                requested_moves = white_requested_moves if board.turn else black_requested_moves

                if i < len(requested_moves):
                    if current_requested_move := requested_moves[i]:
                        move = chess.Move.from_uci(current_requested_move['value'])
                        # Calculate distance moved
                        distance = chess.square_distance(move.from_square, move.to_square)

                        if piece := board.piece_at(move.from_square):
                            piece_type = piece.symbol().lower()

                            # Update total distance and counts
                            if piece_type in distance_data[bot_name]["total_distance"]:
                                distance_data[bot_name]["total_distance"][piece_type] += distance
                                distance_data[bot_name]["counts"][piece_type] += 1
                            else:
                                distance_data[bot_name]["total_distance"][piece_type] = distance
                                distance_data[bot_name]["counts"][piece_type] = 1

# Calculate averages
averages = {}
for bot_name, data in distance_data.items():
    if bot_name not in BOTS_TO_LOG:
        continue
    averages[bot_name] = {}
    for piece_type, total_distance in data["total_distance"].items():
        count = data["counts"][piece_type]
        averages[bot_name][piece_type] = total_distance / count

# Convert the data to a DataFrame for easier plotting with seaborn
plot_data = []
for bot_name, piece_averages in averages.items():
    for piece_type, average in piece_averages.items():
        plot_data.append({'Bot': bot_name, 'Piece Type': piece_type, 'Average Distance Moved': average})
plot_data = pd.DataFrame(plot_data)

# Map abbreviated piece types to full names
piece_name_map = {'p': 'Pawn', 'r': 'Rook', 'n': 'Knight', 'b': 'Bishop', 'q': 'Queen', 'k': 'King'}
plot_data['Piece Type'] = plot_data['Piece Type'].map(piece_name_map)

# Set seaborn style
sns.set(style="whitegrid")

# Create the plot
plt.figure(figsize=(10, 6))
ax = sns.barplot(x='Piece Type', y='Average Distance Moved', hue='Bot', data=plot_data, ci=None, palette='muted')

# Adding labels and title
ax.set_xlabel('Piece Types', fontsize=14)
ax.set_ylabel('Average Distance Moved', fontsize=14)
ax.set_title('Average Distance Moved by Different Piece Types for Each Bot', fontsize=16)
ax.legend(title='Bots', title_fontsize='13', fontsize='11')

# Tweak the visuals a bit
sns.despine(left=True, bottom=True)

# Display the plot
plt.show()
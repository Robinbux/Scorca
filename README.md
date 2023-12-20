# Scorca - A Bot for Reconnaissance Blind Chess

Scorca is an open-source bot designed to play Reconnaissance Blind Chess (RBC). It was created as part of a research project investigating knowledge modelling, sensing strategies, and moving strategies in RBC.

## About
Reconnaissance Blind Chess (RBC) is a variant of chess where players have imperfect information about the state of the board. Each turn, a player can "sense" a 3x3 area of the board to reveal piece locations before moving. The goal is to capture the opponent's king. RBC introduces significant complexity and uncertainty compared to regular chess.
Scorca aims to effectively model knowledge, choose optimal sensing actions, and select strong moves in order to succeed in this uncertain environment. It incorporates techniques such as:

- Entropy-based sensing to maximize information gain
- Heuristics to improve the efficiency of sensing
- Evaluation of move quality across plausible board states
- Utilization of the Leela Chess Zero neural network

The source code can be seen in the `src` directory, while the experiments and plots can be seen in the remaining ones.

During development, Scorca was able to achieve a high ranking on the RBC global leaderboard, demonstrating the effectiveness of its approach. The current Elo can be tracked here: [Scorca Elo Ranking](https://rbc.jhuapl.edu/users/48973)

## Project overview
The main logic can be seen in the files at `src/scorca`. `game_master.py` was used for local tests.
The individual scripts had the following functions:
- `scorca.py` - Main script for running Scorca
- `sense_strategy.py` - Entropy sense strategy
- `move_strategy.py` - Leela based move strategy
- `boards_tracker.py` - Tracker of all possible board states
- `utils.py` - Utility functions

## Configuration Parameters

The behavior of Scorca can be configured using various parameters and hyperparameters. Below is a summary of these settings:

| Parameter | Value | Description |
| --------- | ----- | ----------- |
| `STATES_CUTOFF_POINT` | 10,000 | Maximum number of states to consider |
| `LIKELY_NEXT_STATES_PER_BOARD` | 2 | Number of likely next states per board |
| `OPTIMISTIC_STATES_PER_BOARD` | 0 | Number of optimistic states per board |
| `VALUE_MATE` | 10,000 | Value assigned to a checkmate |
| `TIME_USED_FOR_OPERATION` | 5 | Time used for each operation (in seconds) |
| `KING_CAPTURE_SCORE` | 15 | Score for capturing the king |
| `LEAVE_IN_CHECK_SCORE` | 7 | Score for leaving the opponent in check |
| `CHECK_WITHOUT_CAPTURE_SCORE` | 7 | Score for a check without capture |
| `MAX_BOARDS` | 4,000 | Maximum number of boards to track |
| `SECONDS_PER_PLAYER` | 300 | Time allocated per player (in seconds) |


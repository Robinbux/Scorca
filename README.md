# Scorca - A Bot for Reconnaissance Blind Chess

Scorca is an open-source bot designed to play Reconnaissance Blind Chess (RBC). It was created as part of a research project investigating knowledge modelling, sensing strategies, and moving strategies in RBC.

## About
Reconnaissance Blind Chess (RBC) is a variant of chess where players have imperfect information about the state of the board. Each turn, a player can "sense" a 3x3 area of the board to reveal piece locations before moving. The goal is to capture the opponent's king. RBC introduces significant complexity and uncertainty compared to regular chess.
Scorca aims to effectively model knowledge, choose optimal sensing actions, and select strong moves in order to succeed in this uncertain environment. It incorporates techniques such as:

- Entropy-based sensing to maximize information gain
- Heuristics to improve the efficiency of sensing
- Evaluation of move quality across plausible board states
- Utilization of the Leela Chess Zero neural network

The source coe can be seen in the `src` directory, while the experiments and plots can be seen in the remaining ones.
  
During development, Scorca was able to achieve a high ranking on the RBC global leaderboard, demonstrating the effectiveness of its approach. The current Elo can be tracked bere: https://rbc.jhuapl.edu/users/48973
# CIS667
OthelloBot
For the non ML version:-
Run othellonew.py. To run this we need tkinter

For the ML version:-
First make sure you have downloaded theano, numba, python3, numpy. You can use pip command at the terminal for this.
Make sure you save all the py files in one folder.
Go to that folder using terminal.
Run omegabot.py, optimize.py, othello.py (in that order).
Run train.py
Run game.py

How to play the game (ML version):-
When the game starts enter the name of the spot where you want to put your player. The possible legal moves will be represented as a ‘x’ in the grid. You need to enter the position of your desired position amongst the legal moves. You can type the following things as well:-
’new’		Start a new game
‘x’		Exchange player, enter this at start to play white.
‘exit’	quit the game
‘help’	shows the above mentioned options

How to view the results:-
visualizations(1). Make sure the data file is in the same folder while running. This file is the graphs generated for different learning rates for each team member. To run this we need matplotlib, numpy, itertools, re, glob. The data for this is in the file: CNN_iterations.txt

Command for running these files on terminal: python3 <filename.py>


Note: The codes have been inpsired by:-
1. https://github.com/johnafish/othello
2. https://github.com/khaotik/OmegaOthello

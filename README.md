# Pente-AI
An attempt to create a superhuman AI for playing Pente.

Running pente_zero.py will generate data and train a network on that data.  It runs for a very long time to complete even one iteration due to the amount of data being generated and fed to a neural network.  Running on a GPU will help mitigate this runtime a little bit.

elo_calculator.py can be used to compare models that result from running pente_zero.py

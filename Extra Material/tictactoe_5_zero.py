# -*- coding: utf-8 -*-
"""Copy of MPS_without_value_head.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1ynSip5P5ZhjXbZG7sWPVeXCLfNKd6g--
"""

# !pip install compress_pickle

import numpy as np
import scipy as sp
from scipy.ndimage import correlate
import pandas as pd
import time
import math
import random
import multiprocessing
from datetime import datetime
import sys
import networkx as nx
from collections.abc import Iterable
# import pickle
from random import sample
import matplotlib.pyplot as plt
import seaborn as sns
import compress_pickle as pickle
from keras.layers import Dense, Add, Concatenate, Conv2D, Input, MaxPooling2D, Conv2DTranspose, Flatten, Masking, Reshape, LayerNormalization, Softmax, Activation
from keras import Model
# import tensorflow as tf
import keras
import keras.backend as K
from sympy import Or

class TicTacToe():
  def __init__(self, n_players=2, n_games=100, agent=None, graph_path=None, game_size=3):
    self.game_size = game_size
    self.boards = np.zeros(shape=(n_games, self.game_size, self.game_size)).astype(int)
    self.prev_boards = np.zeros(shape=(n_games, self.game_size, self.game_size)).astype(int)
    self.pairs = np.zeros((n_games, n_players))

    self.agent = agent

    self.n_players = n_players
    self.total_n_games = n_games
    self.n_games = n_games
    self.completed = np.array([False] * n_games)

    self.outcomes = [[],[]]
    
    if graph_path is None:
        self.graph = nx.DiGraph()
    else:
        try:
            with open(graph_path, "rb") as f:
                self.graph = pickle.load(f)
                
            print("----------------------")
            print("LOADED EXISTING GRAPH")
            print("----------------------")
            
        except:
            self.graph = nx.DiGraph()
  
    self.curr_player = 1

    # Create filters for finding surrounded pairs on the board
  
    self.vertical_conv = np.array([[[1],
                                    [1],
                                    [1]]])

    

    self.horizontal_conv = np.array([[[1, 1, 1]]])

    self.main_diag_conv = np.array([[[1, 0, 0],
                                      [0, 1, 0],
                                      [0, 0, 1]]])
    
    self.off_diag_conv = np.array([[[0, 0, 1],
                                    [0, 1, 0],
                                    [1, 0, 0]]])

    print(self.vertical_conv.shape)
    print(self.horizontal_conv.shape)
    print(self.main_diag_conv.shape)
    print(self.off_diag_conv.shape)
    


  def getLegalMoves(self):
    mask = (self.boards == 0)
    return mask

  # def checkRow(self, seq):
  #   winning_seq = np.array([1, 1, 1, 1, 1])
  #   if 5 in np.convolve(winning_seq, seq, "valid"):
  #     return True
  #   else:
  #     return False

  def make_hashable(self, a):
    out = pickle.dumps(a, compression="bz2")
    return out

  def getMoveIndices(self, agent=None, train=True, make_random_moves=False):

    if make_random_moves:
      
      self.values = agent.getValues(self.boards, self.pairs, self.last_player, self.graph, train=train, random_agent=True)
      
      self.mask = self.getLegalMoves()

      self.values += np.random.normal(0, .01, size=self.values.shape)

      self.values *= self.mask

      count = 0

      while np.any(np.sum((np.amax(self.values, axis=(1,2), keepdims=True) == self.values), axis = (1,2)) != 1):

        too_many_choices = self.values[np.sum((np.amax(self.values, axis=(1,2), keepdims=True) == self.values), axis = (1,2)) != 1]

        count += 1

        if too_many_choices.shape[0] != 0:

          rng = np.random.default_rng()
          seq = np.vstack([rng.permutation(self.game_size**2).reshape(1,self.game_size,self.game_size) for i in range(too_many_choices.shape[0])]).astype("float64")
        
          too_many_choices += np.random.normal(0, .25, size=(too_many_choices.shape[0], self.game_size, self.game_size))

          print(seq.shape)

          too_many_choices *= seq

        if count > 100:
          print(too_many_choices)
          sys.exit()


        self.values[np.sum((np.amax(self.values, axis=(1,2), keepdims=True) == self.values), axis = (1,2)) != 1] = too_many_choices

      self.decision_indices = np.swapaxes(np.array((np.argmax(np.max(self.values, axis=2), axis=1), np.argmax(np.max(self.values, axis=1), axis=1))),0,-1)

      decisions = (np.amax(self.values, axis=(1,2), keepdims=True) == self.values)

      return decisions
    
    else:
      # See http://tim.hibal.org/blog/alpha-zero-how-and-why-it-works/ for more information on UCB scores.

      self.values = agent.getValues(self.boards, self.pairs, self.last_player, self.graph, train=train, random_agent=False)
      
      self.mask = self.getLegalMoves()

      # Adding small amount of noise to work around odd issue 
      # where agent tries to take same move over and over.

      self.values += np.random.normal(0, .001, size=self.values.shape)

      self.values *= self.mask

      count = 0

      while np.any(np.sum((np.amax(self.values, axis=(1,2), keepdims=True) == self.values), axis = (1,2)) != 1):
        too_many_choices = self.values[np.sum((np.amax(self.values, axis=(1,2), keepdims=True) == self.values), axis = (1,2)) != 1]

        count += 1

        if too_many_choices.shape[0] != 0:

          rng = np.random.default_rng()
          seq = np.vstack([rng.permutation(self.game_size**2).reshape(1,self.game_size,self.game_size) for i in range(too_many_choices.shape[0])]).astype("float64")
        
          too_many_choices += np.random.normal(0, .25, size=(too_many_choices.shape[0], self.game_size, self.game_size))

          print(seq.shape)

          too_many_choices *= seq

        if count > 100:
          print(too_many_choices)
          sys.exit()

        self.values[np.sum((np.amax(self.values, axis=(1,2), keepdims=True) == self.values), axis = (1,2)) != 1] = too_many_choices

      self.decision_indices = np.swapaxes(np.array((np.argmax(np.max(self.values, axis=2), axis=1), np.argmax(np.max(self.values, axis=1), axis=1))),0,-1)

      decisions = (np.amax(self.values, axis=(1,2), keepdims=True) == self.values)

      return decisions

  def makeMoves(self, player_num, agent, move_num, train=True, make_random_moves=False):

    self.last_player = 0 if player_num == 1 else 1

    # If in training mode, use UCB scores to choose moves, otherwise choose agent predictions
    self.choices = self.getMoveIndices(agent, train=train, make_random_moves=make_random_moves)

    # print(np.sum(self.choices, axis=(1,2)))

    if not np.all(np.isclose(np.sum(self.choices, axis=(1,2)), 1)):
      print(self.choices[np.logical_not(np.isclose(np.sum(self.choices, axis=(1,2)), 1))][0:2,:,:])
      print(self.boards[np.logical_not(np.isclose(np.sum(self.choices, axis=(1,2)), 1))][0:2,:,:])
    assert np.all(np.isclose(np.sum(self.choices, axis=(1,2)), 1))

    # Cache current game state
    self.prev_boards = self.boards

    # print("MAKING MOVES AND UPDATING BOARD")

    if train:
      # Add predecessor nodes to the graph if they aren't already in the graph
      self.last_nodes = [self.make_hashable((data[0], data[1], data[2])) for data in list(zip(self.prev_boards.astype(int), self.pairs[:, self.last_player].astype(int), self.pairs[:, player_num].astype(int)))]

      # print("LAST NODE:\n")
      # print(list(zip(intermediate_boards.tolist(), self.pairs[:, self.last_player].tolist(), self.pairs[:, player_num].tolist())))

      add_last_nodes = [node for node in self.last_nodes if node not in self.graph]
      # print("NUM ORIGINATING NODES NOT IN GRAPH:", len(add_last_nodes))
      self.graph.add_nodes_from(add_last_nodes, visits=1, outcomes=0.5, player=self.last_player)

    # Make moves based on choices
    self.boards += self.choices

    # All possible cases where a player wins by getting three pieces in a row
    case1 = np.sum(correlate(self.boards.astype(np.int16), self.vertical_conv.astype(np.int16), mode="constant") >= 2.5, axis=(1,2))
    case2 = np.sum(correlate(self.boards.astype(np.int16), self.horizontal_conv.astype(np.int16), mode="constant") >= 2.5, axis=(1,2))
    case3 = np.sum(correlate(self.boards.astype(np.int16), self.main_diag_conv.astype(np.int16), mode="constant") >= 2.5, axis=(1,2))
    case4 = np.sum(correlate(self.boards.astype(np.int16), self.off_diag_conv.astype(np.int16), mode="constant") >= 2.5, axis=(1,2))

    print(correlate(self.boards.astype(np.int16), self.vertical_conv.astype(np.int16)).shape)


    # Nested or statements that calculate if any of these cases are True
    finished = np.bitwise_or(np.bitwise_or(np.bitwise_or(case1, case2), case3), case4)
    # print("NUMBER OF GAMES IN PROGRESS:", finished.shape)

    # assert np.all(self.boards >= -.05
    
    # Check if current player won by capturing five or more pairs
    self.completed = np.logical_or(finished, (self.pairs[:, player_num] >= 5))

    # Flip board to be viewed from perspective of other player
    self.boards *= -1

    # print("ADDING RESULTING NODES TO GRAPH")

    if train:
      # Add current nodes to the graph if they aren't already in the graph
      curr_nodes = [self.make_hashable((data[0], data[1], data[2])) for data in list(zip(self.boards.astype(int), self.pairs[:, player_num].astype(int), self.pairs[:, self.last_player].astype(int)))]
      add_curr_nodes = [node for node in curr_nodes if node not in self.graph]
      self.graph.add_nodes_from(add_curr_nodes, visits=1, outcomes=0.5, player=player_num)

      # Add edges between previous game states and corresponding new game state
      edges_with_data = list(zip(self.last_nodes, curr_nodes, self.decision_indices))

      self.graph.add_edges_from([(edge[0], edge[1], {"move":edge[2]}) for edge in edges_with_data])

      # Store node visit count and outcome (won by player 1 or 0) in graph
      completed_nodes_data = list(zip(self.boards[self.completed].astype(int), self.pairs[:, player_num][self.completed].astype(int), self.pairs[:, self.last_player][self.completed].astype(int)))
      completed_nodes = [self.make_hashable((data[0], data[1], data[2])) for data in completed_nodes_data]

      for node in completed_nodes:
        if move_num < 3:
          print(self.completed)
          print(self.boards[self.completed])
          sys.exit()
        ancestors = nx.ancestors(self.graph, node)
        # print("NUM ANCESTORS BEING UPDATED:", len(ancestors))
        for node in ancestors:
          self.graph.nodes[node]["visits"] += 1
          self.graph.nodes[node]["outcomes"] += int(player_num == self.graph.nodes[node]["player"])

    # Update outcome records
    self.outcomes[player_num] += [move_num]*sum(self.completed)

    # Drop all completed games from current computation
    self.boards = self.boards[np.logical_not(self.completed)]
    self.pairs = self.pairs[np.logical_not(self.completed)]
    self.n_games = self.pairs.shape[0]

    if self.n_games != self.total_n_games:
      self.completed = np.array([False] * self.n_games)
    # print("SHAPE OF CURRENT BOARDS MATRIX AFTER DROPPING COMPLETED GAMES:", self.boards.shape)
### 
          
    return self.completed  

  def getBoards(self):
    return self.boards

  def play_train(self, agent0, agent1, make_random_moves=[False, False], num_games=100):

    self.total_n_games = num_games
    self.reset()

    count = 0
    player = -1
    while not np.sum(self.completed) == self.n_games and count < self.game_size ** 2:

      if player < self.n_players - 1:
        player+=1
      else:
        player=0

      if make_random_moves[0]:
        if player==0:
          fin_list = self.makeMoves(player, agent0, count, make_random_moves=True) 
      else:
        if player==0:
          fin_list = self.makeMoves(player, agent0, count)

      if make_random_moves[1]:
        if player==1:
          fin_list = self.makeMoves(player, agent1, count, make_random_moves=True)
      else:
        if player==1:
          fin_list = self.makeMoves(player, agent1, count)

      print("TAKING TURN", count)
      count += 1

    print("DONE PLAYING", self.total_n_games, "PENTE GAMES")

    # agent0.fit_on_graph(self.graph)

    print("==================")
    print("Player 0 won ", len(self.outcomes[0])/self.total_n_games * 100, "percent of games")
    print("Player 1 won ", len(self.outcomes[1])/self.total_n_games * 100, "percent of games")
    print("==================")    

    self.reset()

  def play_test(self, agent0, agent1, make_random_moves=[False, False], num_games=100):

    self.total_n_games = num_games
    self.reset()

    count = 0
    player = -1
    while not np.sum(self.completed) == self.n_games and count < self.game_size ** 2:

      if player < self.n_players - 1:
        player+=1
      else:
        player=0

      if make_random_moves[0]:
        if player==0:
          fin_list = self.makeMoves(player, agent0, count, make_random_moves=True) 
      else:
        if player==0:
          fin_list = self.makeMoves(player, agent0, count)

      if make_random_moves[1]:
        if player==1:
          fin_list = self.makeMoves(player, agent1, count, make_random_moves=True)
      else:
        if player==1:
          fin_list = self.makeMoves(player, agent1, count)

      print("TAKING TURN", count)
      count += 1

    print("DONE PLAYING", self.total_n_games, "PENTE GAMES")

    # agent0.fit_on_graph(self.graph)

    print("=========TEST GAMES=========")
    print("Player 0 won ", len(self.outcomes[0])/self.total_n_games * 100, "percent of games")
    print("Player 1 won ", len(self.outcomes[1])/self.total_n_games * 100, "percent of games")
    print("=========TEST GAMES=========")    

    outcome0, outcome1 = len(self.outcomes[0])/self.total_n_games * 100, len(self.outcomes[1])/self.total_n_games * 100
    self.reset()

    return outcome0, outcome1

  def reset(self):
    self.n_games = self.total_n_games
    self.boards = np.zeros(shape=(self.n_games, 5, 5))
    self.pairs = np.zeros((self.n_games, self.n_players))
    self.completed = [False for i in range(self.n_games)]
    self.curr_player = 1
    self.outcomes = [[],[]]

class Agent():
  def __init__(self, agent_path = None, intelligence = 1):

    self.node_values = {}
    self.scale_factor = 1

    if agent_path is None:
      inp1 = Input(shape=(5,5,1))
      inp_mask = Input(shape=(5,5,1))

      # Convolve
      conv1 = Conv2D(512, 2, activation="elu")(inp1)
      # norm1 = LayerNormalization()(conv1)

      deconv1 = Conv2DTranspose(256, 2, activation="elu")(conv1)
      # norm2 = LayerNormalization()(deconv1)

      conv2 = Conv2D(128, 2, activation="elu")(deconv1)
      # norm3 = LayerNormalization()(conv2)

      policy_head = Conv2DTranspose(1, 2, activation="sigmoid")(conv2)
      # norm4 = LayerNormalization()(deconv2)

      # # add = Add()([norm4 * inp_mask, -1*K.cast(inp_mask==0, "float32")])

      # # # mask = Masking(mask_value=-1)(add)

      # # mask = K.not_equal(add, -1)

      # conv3 = Conv2D(64, 2, activation="relu")(norm4)
      # norm5 = LayerNormalization()(conv3)

      # deconv3 = Conv2DTranspose(16, 2, activation="relu")(norm5)
      # norm6 = LayerNormalization()(deconv3)

      # conv4 = Conv2D(8, 2, activation="relu")(norm6)

      #  = Conv2DTranspose(1, 2, activation="sigmoid", name="policy_head")(conv4)

      # mask = K.not_equal(add, -1)

      # policy_head = keras.layers.Softmax(name="policy_head")(norm6, mask=mask)

      # add = Add()([norm6 * inp_mask, -1*K.cast(inp_mask==0, "float32")])

      # policy_head = Masking(mask_value=-1)(add)

      # mask = K.not_equal(add, -1)

      # policy_head = Activation("sigmoid", name="policy_head")(mask)

      # # Network head for the value of the current (parent) game state
      # dense5 = Dense(256, activation="sigmoid")(dense3)
      # norm11 = LayerNormalization()(dense5)
      # dense6 = Dense(128, activation="sigmoid")(norm11)
      # value_head = Dense(1, activation="sigmoid", name="value_head")(dense6)

      self.model = Model([inp1, inp_mask], policy_head)

      # Scale up loss to make changes more easily visible during training
      # self.model.compile(loss=lambda y_true, y_pred: keras.losses.MeanAbsoluteError()(y_true, y_pred), optimizer="rmsprop")
      self.model.compile(loss="mae", optimizer="adam")
      self.model.summary()
    else:
      self.model = keras.models.load_model(agent_path)

    # Hyperparameter for MCTS tuning
    self.c = .5

  def fit_on_graph(self, graph, num_epochs = 25, batch_size = 256, sample_size = 1000):

    for epoch in range(num_epochs):

      boards = []
      pairs = []
      masks = []
      outcome_train = []
      values = []


      for data in sample(graph.nodes(data=True), sample_size):
        node = data[0]

        if node in self.node_values.keys():

          outcomes = data[1]["outcomes"]
          visits = data[1]["visits"]
          player_num = data[1]["player"]
          value = outcomes / visits if visits != 0 else .1

          (board, cur_pairs, prev_pairs) = pickle.loads(node, compression="bz2")

          l = [(edge[0], tuple(edge[2]["move"]), self.get_node_value(graph, edge[1]), graph.nodes(data=True)[edge[1]]["visits"]) for edge in list(graph.out_edges(node, data=True))]

          for data in l:

            if data[0] not in self.node_values.keys():

              self.node_values[data[0]] = (np.zeros(shape=(5,5)), np.zeros(shape=(5,5))), np.zeros(shape=(5,5))
              
              # print("DATA[1]", data[1])


            # Add outcome score for child
            self.node_values[data[0]][0][data[1]] = data[2]
            # Add 1 to indicate that a node has been visited for masking training later
            self.node_values[data[0]][1][data[1]] = 1
            # Update visit count for particular child node
            self.node_values[data[0]][2][data[1]] = data[3]

          if np.any(self.node_values[node][2] != 0):

            masks.append(self.node_values[node][1])
            outcome_train.append(self.scale_factor * self.node_values[node][0])
            boards.append(board)
            pairs.append([cur_pairs, prev_pairs])
            values.append([value])

      masks = np.array(masks)
      pairs = np.array(pairs)
      values = np.array(values)
      outcome_train = np.array(outcome_train)
      outcome_train = np.reshape(outcome_train, (outcome_train.shape[0], 5, 5, 1))

      assert not np.any(outcome_train > 1)
      assert not np.any(outcome_train < 0)

      # Replace all nan values with 0 that resulted by dividing by zero earlier
      # print(np.sum(outcome_train[~np.isnan(outcome_train)]))
      # t = outcome_train[~np.isnan(outcome_train)].shape[0]*3*3
      # print(np.sum(outcome_train[~np.isnan(outcome_train)]) / t)
      # sys.exit()
      outcome_train[np.isnan(outcome_train)] = 0

      boards = np.array(boards)



      # Train only on data where "masks" has some non-zero elements (i.e. has some explored positions)
      temp = (np.sum(masks, axis=(1,2)) != 0)

      if np.sum(temp) != 0:
          
        temp_masks = masks[temp, :, :]
        temp_boards = boards[temp, :, :]
        temp_outcome = outcome_train[temp, :, :, :]
        temp_values = values[temp, :]
        temp_pairs = pairs[temp, :]

        # Data augmentation with horizontal/vertical flip of boards and transposing

        temp_masks = np.vstack((temp_masks, np.flip(temp_masks, axis=1), 
                                np.transpose(np.flip(temp_masks, axis=1), (0,2,1)), 
                                np.transpose(temp_masks, (0,2,1)), np.flip(temp_masks, axis=2), 
                                np.transpose(np.flip(temp_masks, axis=2), (0,2,1)), 
                                np.transpose(np.flip(np.flip(temp_masks, axis=2), axis=1), (0,2,1)), 
                                np.flip(np.flip(temp_masks, axis=2), axis=1)))
        
        temp_boards = np.vstack((temp_boards, np.flip(temp_boards, axis=1), 
                                np.transpose(np.flip(temp_boards, axis=1), (0,2,1)), 
                                np.transpose(temp_boards, (0,2,1)), np.flip(temp_boards, axis=2), 
                                np.transpose(np.flip(temp_boards, axis=2), (0,2,1)), 
                                np.transpose(np.flip(np.flip(temp_boards, axis=2), axis=1), (0,2,1)), 
                                np.flip(np.flip(temp_boards, axis=2), axis=1)))
        print(temp_outcome.shape)
        temp_outcome = np.vstack((temp_outcome, np.flip(temp_outcome, axis=1), 
                                np.transpose(np.flip(temp_outcome, axis=1), (0,2,1,3)), 
                                np.transpose(temp_outcome, (0,2,1,3)), np.flip(temp_outcome, axis=2), 
                                np.transpose(np.flip(temp_outcome, axis=2), (0,2,1,3)), 
                                np.transpose(np.flip(np.flip(temp_outcome, axis=2), axis=1), (0,2,1,3)), 
                                np.flip(np.flip(temp_outcome, axis=2), axis=1)))
        
        temp_values = np.repeat(temp_values, 8, axis=0)
        temp_pairs = np.repeat(temp_pairs, 8, axis=0)

        temp_boards = np.vstack((temp_boards, -1*temp_boards))
        temp_masks = np.repeat(temp_masks, 2, axis=0)
        temp_values = np.vstack((temp_values, 1-temp_values))
        temp_pairs = np.vstack((temp_pairs, np.flip(temp_pairs, axis=1)))
        temp_outcome = np.vstack((temp_outcome, self.scale_factor-temp_outcome))

        assert not np.isnan(temp_boards).any()
        assert not np.isnan(temp_masks).any()
        assert not np.isnan(temp_outcome).any()
        assert not np.any(temp_outcome > 1)
        assert not np.any(temp_outcome < 0)
          
        self.model.fit([temp_boards, temp_masks], temp_outcome, epochs=1, batch_size=batch_size, shuffle=True, validation_split=.1)

  def get_node_value(self, graph, node):
    if graph.nodes(data=True)[node]["visits"] != 0:
      return graph.nodes(data=True)[node]["outcomes"] / graph.nodes(data=True)[node]["visits"]
    else:
      return np.inf

  def getValues(self, boards, pairs, player_num, graph, train=True, random_agent=False):

    n = boards.shape[0]
    # print(board_states.shape)

    next_player = 0 if player_num == 1 else 1
    players_list = [player_num] * n
    curr_pairs = pairs[:, next_player].astype(int)
    last_pairs = pairs[:, player_num].astype(int)
    inp2 = np.array(list(zip(curr_pairs, last_pairs)))

    # # For inference, do not mask out anything, unlike when training where we mask out unexplored moves
    # # Masking out illegal moves will be done later
    pred_masks = np.ones((n, 5, 5, 1))

    if train:
      node_list = [multi.make_hashable((x[0], x[1], x[2])) for x in list(zip(boards.astype(int), last_pairs, curr_pairs))]

      l = [(edge[0], tuple(edge[2]["move"]), self.get_node_value(graph, edge[1]), graph.nodes(data=True)[edge[1]]["visits"]) for edge in list(graph.out_edges(node_list, data=True))]
      # print(len(l))
      l += [(node, None, 0) for node in node_list]

      for data in l:
        if data[0] not in self.node_values.keys():

          self.node_values[data[0]] = (np.zeros(shape=(5,5)), np.zeros(shape=(5,5)), np.zeros(shape=(5,5)))
          
          if data[1] is not None:
            # Add outcome score for child
            self.node_values[data[0]][0][data[1]] = data[2]
            # Add 1 to indicate that a node has been visited for masking training later
            self.node_values[data[0]][1][data[1]] = 1
            # Update visit count for particular child node
            self.node_values[data[0]][2][data[1]] = data[3]

        else:
          if data[1] is not None:
            # Add outcome score for child
            self.node_values[data[0]][0][data[1]] = data[2]
            # Add 1 to indicate that a node has been visited for masking training later
            self.node_values[data[0]][1][data[1]] = 1
            # Update visit count for particular child node
            self.node_values[data[0]][2][data[1]] = data[3]

      if not random_agent:

        parent_count_array = np.array([graph.nodes(data=True)[node]["visits"] if node in graph else 0 for node in node_list]).reshape((n, 1, 1, 1))
        child_count_array = np.array([self.node_values[node][2] for node in node_list]).reshape((n, 5, 5, 1))
        outcome_array = np.array([self.node_values[node][0] for node in node_list]).reshape((n, 5, 5, 1))

        policy_preds = self.model.predict([boards, pred_masks])

        print(policy_preds.shape)

        ucb_score = np.array([outcome_array + self.c * (policy_preds / self.scale_factor) * np.sqrt(np.divide(np.log(parent_count_array + .0001), child_count_array + .000000001))]).reshape((n, 5, 5))

        # ucb_score[np.isnan(ucb_score)] = 10000000
        np.nan_to_num(ucb_score, copy=False, nan=10000000, posinf=10000000, neginf=-10000000)

        # print("FINITE UCB_SCORE COUNT:", np.sum((ucb_score != 10000000)))

        
        return ucb_score
      
      else:

        return np.random.uniform(0,1,size=(n, 5, 5))
      
    else:

      if not random_agent:

        policy_preds = self.model.predict([boards, pred_masks])
        np.nan_to_num(policy_preds, copy=False, nan=10000000, posinf=10000000, neginf=-10000000)
        return policy_preds.reshape(n, 5, 5)

      else:

        return np.random.uniform(0,1,size=(n, 5, 5))

agent0 = Agent()

multi = TicTacToe(n_games = 1000, game_size=5)

num_batches = 10

path = "graph.pkl"

# Start with random playouts to train on
for batch in range(1):
  multi.play_train(agent0, agent0, make_random_moves=[True, True], num_games=10000)

num_batches = 50

agent_win_rate = []
batches = list(range(num_batches))

# Then add "intelligent" agent playouts
for batch in range(num_batches):

  print("BATCH", batch)
  multi.play_train(agent0, agent0, num_games=100000)

  print("===============")
  print("GRAPH DETAILS")
  print(multi.graph)
  print("===============")

  agent0.fit_on_graph(multi.graph, num_epochs=(batch + 1), batch_size=4096, sample_size=30000)

  zerowinrate, onewinrate = multi.play_test(agent0, agent0, make_random_moves=[False, True], num_games=1000)
  agent_win_rate.append(zerowinrate)

  # zerowinrate, onewinrate = multi.play_test(agent0, agent0, make_random_moves=[False, True], num_games=1000)

  # if batch % 10 == 1:
  #   keras.models.save_model(agent0.model, f"drive/MyDrive/pente_ai.h5")

# with open(path, "wb") as f:
#   pickle.dump(multi.graph, f)

# keras.models.save_model(agent0.model, f"drive/MyDrive/pente_ai.h5")
# agent0.fit_on_graph(multi.graph, num_epochs=1000)
# keras.models.save_model(agent0.model, f"drive/MyDrive/pente_ai.h5")

# nx.draw(multi.graph, node_size=10)

# roots = {n for n,d in multi.graph.in_degree() if d==0}

# len(roots)

plt.scatter(x=batches, y=agent_win_rate)
plt.ylim([0,1])
plt.show()


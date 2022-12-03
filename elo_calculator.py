import numpy as np
import os
from keras.models import load_model
import random
from scipy.ndimage import correlate
import matplotlib.pyplot as plt

class Pente():
  def __init__(self, n_players=2, n_games=100, model=None, graph_path=None):
    self.boards = np.zeros(shape=(n_games, 19, 19)).astype(int)
    self.prev_boards = np.zeros(shape=(n_games, 19, 19)).astype(int)
    self.pairs = np.zeros((n_games, n_players))

    self.model = model

    self.n_players = n_players
    self.total_n_games = n_games
    self.n_games = n_games
    self.completed = np.array([False] * n_games)

    self.outcomes = [[],[]]
  
    self.curr_player = 1

    # Create filters for finding surrounded pairs on the board
    self.vertical_pairs = np.array([[[1],
                                    [-1],
                                    [-1],
                                    [1]]])

    self.horizontal_pairs = np.array([[[1, -1, -1, 1]]])

    self.main_diag_pairs = np.array([[[1, 0, 0, 0],
                                      [0, -1, 0, 0],
                                      [0, 0, -1, 0],
                                      [0, 0, 0, 1]]])

    self.off_diag_pairs = np.array([[[0, 0, 0, 1],
                                    [0, 0, -1, 0],
                                    [0, -1, 0, 0],
                                    [1, 0, 0, 0]]])
    
    self.vertical_conv = np.array([[[1],
                                    [1],
                                    [1],
                                    [1],
                                    [1]]])

    self.horizontal_conv = np.array([[[1, 1, 1, 1, 1]]])

    self.main_diag_conv = np.array([[[1, 0, 0, 0, 0],
                                      [0, 1, 0, 0, 0],
                                      [0, 0, 1, 0, 0],
                                      [0, 0, 0, 1, 0],
                                      [0, 0, 0, 0, 1]]])
    
    self.off_diag_conv = np.array([[[0, 0, 0, 0, 1],
                                    [0, 0, 0, 1, 0],
                                    [0, 0, 1, 0, 0],
                                    [0, 1, 0, 0, 0],
                                    [1, 0, 0, 0, 0]]])
    
  def getLegalMoves(self):
    mask = (self.boards == 0)
    return mask

  def getMoveIndices(self, model, make_random_moves=False):

    if make_random_moves:
      
      self.values = np.random.normal(.5, .1, size=self.values.shape)
      
      self.mask = self.getLegalMoves()

      self.values += self.mask


      rng = np.random.default_rng()
      seq = np.vstack([rng.permutation(361).reshape(1,19,19) for i in range(self.values.shape[0])])

      # self.decision_indices = np.swapaxes(np.array((np.argmax(np.max(self.values, axis=2), axis=1), np.argmax(np.max(self.values, axis=1), axis=1))),0,-1)

      self.values = (np.amax(self.values, axis=(1,2), keepdims=True) == self.values).astype(int)
      self.values += seq
      decisions = (np.amax(self.values, axis=(1,2), keepdims=True) == self.values)

      return decisions
    
    else:
      # See http://tim.hibal.org/blog/alpha-zero-how-and-why-it-works/ for more information on UCB scores.
      
      self.values = model.predict([self.boards, self.pairs])
      decisions = np.ones(shape=self.values.shape)

      while not np.all(np.isclose(np.sum(decisions, axis=(1,2)), 1)):
        
        self.mask = self.getLegalMoves()

        # Adding small amount of noise to work around odd issue 
        # where model tries to take same move over and over.

        self.values += np.random.normal(0, .005, size=self.values.shape)

        if self.values.shape[-1] == 1:
          self.values = np.squeeze(self.values, axis=-1)

        self.values *= self.mask

        rng = np.random.default_rng()
        seq = np.vstack([rng.permutation(361).reshape(1,19,19) for i in range(self.values.shape[0])])

        # self.values = (np.amax(self.values, axis=(1,2), keepdims=True).astype("float64") == self.values.astype("float64")).astype(int)
        # self.values += seq
        decisions = (np.amax(self.values, axis=(1,2), keepdims=True).astype("float64") == self.values.astype("float64"))

      return decisions

  def makeMoves(self, player_num, model, move_num, make_random_moves=False):

    self.last_player = 0 if player_num == 1 else 1

    # If in training mode, use UCB scores to choose moves, otherwise choose model predictions
    self.choices = self.getMoveIndices(model, make_random_moves=make_random_moves)

    if not np.all(np.isclose(np.sum(self.choices, axis=(1,2)), 1)):
      print(self.choices[np.logical_not(np.isclose(np.sum(self.choices, axis=(1,2)), 1))])
      print(self.boards[np.logical_not(np.isclose(np.sum(self.choices, axis=(1,2)), 1))])
    assert np.all(np.isclose(np.sum(self.choices, axis=(1,2)), 1))

    # Cache current game state
    self.prev_boards = self.boards

    # Make moves based on choices
    self.boards += self.choices

    # All possible cases where a player wins by getting five pieces in a row
    case1 = np.sum(correlate(self.boards, self.vertical_conv, mode="constant") >= 4.5, axis=(1,2))
    case2 = np.sum(correlate(self.boards, self.horizontal_conv, mode="constant") >= 4.5, axis=(1,2))
    case3 = np.sum(correlate(self.boards, self.main_diag_conv, mode="constant") >= 4.5, axis=(1,2))
    case4 = np.sum(correlate(self.boards, self.off_diag_conv, mode="constant") >= 4.5, axis=(1,2))

    # Nested or statements that calculate if any of these cases are True
    finished = np.bitwise_or(np.bitwise_or(np.bitwise_or(case1, case2), case3), case4)

    # Find pairs that existed on board prior to moving
    prev_vert_pairs = (correlate(self.prev_boards, self.vertical_pairs, mode="constant") >= 3.5)
    prev_hor_pairs = (correlate(self.prev_boards, self.horizontal_pairs, mode="constant") >= 3.5)
    prev_main_pairs = (correlate(self.prev_boards, self.main_diag_pairs, mode="constant") >= 3.5)
    prev_off_pairs = (correlate(self.prev_boards, self.off_diag_pairs, mode="constant") >= 3.5)

    # Find pairs that exist on board after making moves
    curr_vert_pairs = (correlate(self.boards, self.vertical_pairs, mode="constant") >= 3.5)
    curr_hor_pairs = (correlate(self.boards, self.horizontal_pairs, mode="constant") >= 3.5)
    curr_main_pairs = (correlate(self.boards, self.main_diag_pairs, mode="constant") >= 3.5)
    curr_off_pairs = (correlate(self.boards, self.off_diag_pairs, mode="constant") >= 3.5)

    # Find ONLY the pairs that resulted directly from making the most recent moves
    delta_vert_pairs = np.logical_xor(curr_vert_pairs, prev_vert_pairs)
    delta_hor_pairs = np.logical_xor(curr_hor_pairs, prev_hor_pairs)
    delta_main_pairs = np.logical_xor(curr_main_pairs, prev_main_pairs)
    delta_off_pairs = np.logical_xor(curr_off_pairs, prev_off_pairs)

    remove_vert = np.logical_or(delta_vert_pairs, np.roll(delta_vert_pairs, -1, axis=1))
    remove_hor = np.logical_or(delta_hor_pairs, np.roll(delta_hor_pairs, -1, axis=2))
    remove_main = np.logical_or(delta_main_pairs, np.roll(np.roll(delta_main_pairs, -1, axis=2), -1, axis=1))
    remove_off = np.logical_or(np.roll(delta_off_pairs, -1, axis=1), np.roll(delta_off_pairs, -1, axis=2))

    remove_all = remove_vert.astype(int) + remove_hor.astype(int) + remove_main.astype(int) + remove_off.astype(int)

    self.boards -= remove_all

    # Update pair count
    self.pairs[:, player_num] += np.sum(remove_all, axis=(1,2)) / 2
    
    # Check if current player won by capturing five or more pairs
    self.completed = np.logical_or(finished, (self.pairs[:, player_num] >= 5))

    # Flip board to be viewed from perspective of other player
    self.boards *= -1

    # Update outcome records
    self.outcomes[player_num] += [move_num]*sum(self.completed)

    # Drop all completed games from current computation
    self.boards = self.boards[np.logical_not(self.completed)]
    self.pairs = self.pairs[np.logical_not(self.completed)]
    self.n_games = self.pairs.shape[0]

    if self.n_games != self.total_n_games:
      self.completed = np.array([False] * self.n_games)
    
    return self.completed  

  def getBoards(self):
    return self.boards

  def play(self, model0, model1, make_random_moves=[False, False], num_games=100):

    self.total_n_games = num_games
    self.reset()

    count = 0
    player = -1
    while not np.sum(self.completed) == self.n_games and count < 299:

      if player < self.n_players - 1:
        player+=1
      else:
        player=0

      if make_random_moves[0]:
        if player==0:
          fin_list = self.makeMoves(player, model0, count, make_random_moves=True) 
      else:
        if player==0:
          fin_list = self.makeMoves(player, model0, count)

      if make_random_moves[1]:
        if player==1:
          fin_list = self.makeMoves(player, model1, count, make_random_moves=True)
      else:
        if player==1:
          fin_list = self.makeMoves(player, model1, count)

      print("TAKING TURN", count)
      count += 1

    print("DONE PLAYING", self.total_n_games, "PENTE GAMES")

    print("==================")
    print("Player 0 won ", len(self.outcomes[0])/self.total_n_games * 100, "percent of games")
    print("Player 1 won ", len(self.outcomes[1])/self.total_n_games * 100, "percent of games")
    print("==================")    
    
    outcome0 = len(self.outcomes[0])/self.total_n_games * 100
    outcome1 = len(self.outcomes[1])/self.total_n_games * 100

    self.reset()

    return outcome0, outcome1

  def reset(self):
    self.n_games = self.total_n_games
    self.boards = np.zeros(shape=(self.n_games, 19, 19))
    self.pairs = np.zeros((self.n_games, self.n_players))
    self.completed = [False for i in range(self.n_games)]
    self.curr_player = 1
    self.outcomes = [[],[]]


elo_dict = {}
model_dict = {}


### EDIT THESE PARAMETERS TO CHANGE ELO UPDATE BEHAVIOR
L = 50
num_update_iters = 100000

k = 25  # Can be scaled with higher or lower elo, but will leave like this
###



for d in os.listdir("models"):
  if os.path.isdir(os.path.join("models",d)):
    if int(d.split("_")[-1]) in [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]:
      elo_dict[os.path.join("models",d)] = 100

for player in list(elo_dict.keys()):
  print(player)
  model_dict[player] = load_model(player)


# elo_dict["random"] = 100

pente = Pente()

for i in range(num_update_iters):

  player = random.choice(list(elo_dict.keys()))
  player2 = random.choice(list(elo_dict.keys()))

  model1 = model_dict[player]
  model2 = model_dict[player2]

  zerowinrate, onewinrate = pente.play(model1, model2, num_games=100)

  rating1_prev = elo_dict[player]
  rating2_prev = elo_dict[player2]

  expected_outcome1 = 1 / (1 + 10**((rating2_prev - rating1_prev)/400))
  expected_outcome2 = 1 / (1 + 10**((rating1_prev - rating2_prev)/400))

  new_rating1 = max(rating1_prev + k * (zerowinrate / 100 - expected_outcome1),100)
  new_rating2 = max(rating2_prev + k * (onewinrate / 100 - expected_outcome2),100)
  
  elo_dict[player] = new_rating1
  elo_dict[player2] = new_rating2

  if i % 50 == 0:
    player_nums = [int(player.split("_")[-1]) for player in elo_dict.keys()]
    ratings = [elo_dict[f"models/pente_ai_{x}"] for x in player_nums]

    plt.scatter(x=player_nums, y=ratings)
    plt.savefig(f"plots/train_history_{i}")
    plt.clf()

  print("======================")
  print("======================")
  print("RUNNING ITERATION: ", i)
  print("======================")
  print("======================")




    
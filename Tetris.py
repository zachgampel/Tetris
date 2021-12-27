import pygame
import pygame.locals
import sys

pygame.init()
from random import random, randint
import numpy as np
from math import exp, ceil
from copy import deepcopy
import os
import re


class Tile():
    def __init__(self):
        self.occupied = False
        self.color = None
        self.current_piece = False


class Board():
    def __init__(self, rows, hidden_rows, cols, tile_length):
        self.rows = rows
        self.hidden_rows = hidden_rows
        self.cols = cols
        self.tile_length = tile_length

        self.current_piece_locations = []
        self.board = []
        for row in range(self.rows):
            r = []
            for col in range(self.cols):
                r.append(Tile())
            self.board.append(r)

    def clear(self):
        self.board = []
        for row in range(self.rows):
            r = []
            for col in range(self.cols):
                r.append(Tile())
            self.board.append(r)

    def draw(self, window):
        for i in range(self.rows + 1):
            pygame.draw.line(window, (255, 255, 255), [0, self.tile_length * i],
                             [self.tile_length * self.cols, self.tile_length * i])
        for j in range(self.cols + 1):
            pygame.draw.line(window, (255, 255, 255), [self.tile_length * j, 0],
                             [self.tile_length * j, self.tile_length * self.rows])

        for i in range(self.rows):
            for j in range(self.cols):
                tile = self.board[i][j]
                if tile.occupied:
                    pygame.draw.rect(window, tile.color,
                                     [j * self.tile_length, (i - self.hidden_rows) * self.tile_length, self.tile_length,
                                      self.tile_length])

    def add_moving_piece(self, piece):
        for i in range(4):
            for j in range(4):
                if piece.positions[i][j] == 1:
                    self.board[i + piece.origin[0]][j + piece.origin[1]].occupied = True
                    self.board[i + piece.origin[0]][j + piece.origin[1]].current_piece = True
                    self.board[i + piece.origin[0]][j + piece.origin[1]].color = piece.color

                    self.current_piece_locations.append([i + piece.origin[0], j + piece.origin[1]])

    def remove_current_piece(self):
        for [i, j] in self.current_piece_locations:
            self.board[i][j].occupied = False
            self.board[i][j].current_piece = False

        self.current_piece_locations = []

    def check_if_piece_alive(self, piece):
        piece.drop(self)
        piece_stays_alive = not self.check_for_collision(piece)
        piece.undrop(self)

        return piece_stays_alive

    def check_for_collision(self, piece):
        for i in range(4):
            for j in range(4):
                if piece.positions[i][j] == 1:
                    if i + piece.origin[0] < 0 or i + piece.origin[0] >= self.rows or j + piece.origin[1] < 0 or j + \
                            piece.origin[1] >= self.cols:
                        return True
                    if self.board[i + piece.origin[0]][j + piece.origin[1]].occupied == True:
                        return True
        return False

    def add_piece(self, piece):
        for i in range(4):
            for j in range(4):
                if piece.positions[i][j] == 1:
                    self.board[i + piece.origin[0]][j + piece.origin[1]].occupied = True
                    self.board[i + piece.origin[0]][j + piece.origin[1]].color = piece.color

    def clear_lines(self):
        lines_cleared = 0
        for i, row in enumerate(self.board[::-1]):
            occupied_count = 0
            for tile in row:
                if tile.occupied:
                    occupied_count += 1
            if occupied_count == self.cols:
                lines_cleared += 1
                self.board.remove(row)

        for row in range(lines_cleared):
            r = []
            for col in range(self.cols):
                r.append(Tile())
            self.board.insert(0, r)

        return lines_cleared

    def check_for_gameover(self, piece):
        return self.check_for_collision(piece)
    
    def get_board_information_as_2d_array(self, live_piece, exclude_hidden_rows=True):
        results = np.zeros((self.rows, self.cols), int)
        
        for i, row in enumerate(self.board):
            for j, tile in enumerate(row):
                if tile.occupied:
                    results[i][j] = 1
        
        for i in range(4):
            for j in range(4):
                if live_piece.positions[i][j]:
                    results[i + live_piece.origin[0]][j + live_piece.origin[1]]
        
        if exclude_hidden_rows:
            results = results[self.hidden_rows:]
        
        return results


class Piece():
    def __init__(self, evolution, replay_info):
        if evolution.active:
            self.shape = randint(0, 6)
            evolution.update_rng(self)
        elif replay_info.active:
            self.shape = replay_info.get_next_rng()
        else:
            self.shape = randint(0, 6)
            
        self.orientation = 0
        self.origin = self.get_origin()
        self.positions = self.get_positions()
        self.color = self.get_color()

    def rotate_clockwise(self, board):
        self.orientation = (self.orientation + 1) % 4
        self.positions = self.get_positions()

        if board.check_for_collision(self):
            self.orientation = (self.orientation - 1) % 4
            self.positions = self.get_positions()
        return self

    def rotate_counterclockwise(self, board):
        self.orientation = (self.orientation - 1) % 4
        self.positions = self.get_positions()

        if board.check_for_collision(self):
            self.orientation = (self.orientation + 1) % 4
            self.positions = self.get_positions()

        return self

    def get_positions(self):
        if self.shape == 0:  # I piece
            if self.orientation == 0:
                return [[0, 0, 0, 0],
                        [1, 1, 1, 1],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0]]
            if self.orientation == 1:
                return [[0, 0, 1, 0],
                        [0, 0, 1, 0],
                        [0, 0, 1, 0],
                        [0, 0, 1, 0]]
            if self.orientation == 2:
                return [[0, 0, 0, 0],
                        [1, 1, 1, 1],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0]]
            if self.orientation == 3:
                return [[0, 1, 0, 0],
                        [0, 1, 0, 0],
                        [0, 1, 0, 0],
                        [0, 1, 0, 0]]
        if self.shape == 1:  # J piece
            if self.orientation == 0:
                return [[0, 0, 0, 0],
                        [1, 1, 1, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 0]]
            if self.orientation == 1:
                return [[0, 1, 0, 0],
                        [0, 1, 0, 0],
                        [1, 1, 0, 0],
                        [0, 0, 0, 0]]
            if self.orientation == 2:
                return [[0, 0, 0, 0],
                        [1, 0, 0, 0],
                        [1, 1, 1, 0],
                        [0, 0, 0, 0]]
            if self.orientation == 3:
                return [[0, 1, 1, 0],
                        [0, 1, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 0, 0]]
        if self.shape == 2:  # L piece
            if self.orientation == 0:
                return [[0, 0, 0, 0],
                        [1, 1, 1, 0],
                        [1, 0, 0, 0],
                        [0, 0, 0, 0]]
            if self.orientation == 1:
                return [[1, 1, 0, 0],
                        [0, 1, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 0, 0]]
            if self.orientation == 2:
                return [[0, 0, 0, 0],
                        [0, 0, 1, 0],
                        [1, 1, 1, 0],
                        [0, 0, 0, 0]]
            if self.orientation == 3:
                return [[0, 1, 0, 0],
                        [0, 1, 0, 0],
                        [0, 1, 1, 0],
                        [0, 0, 0, 0]]
        if self.shape == 3:  # square shape
            return [[0, 0, 0, 0],
                    [0, 1, 1, 0],
                    [0, 1, 1, 0],
                    [0, 0, 0, 0]]
        if self.shape == 4:  # S piece
            if self.orientation == 0:
                return [[0, 0, 0, 0],
                        [0, 1, 1, 0],
                        [1, 1, 0, 0],
                        [0, 0, 0, 0]]
            if self.orientation == 1:
                return [[1, 0, 0, 0],
                        [1, 1, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 0, 0]]
            if self.orientation == 2:
                return [[0, 0, 0, 0],
                        [0, 1, 1, 0],
                        [1, 1, 0, 0],
                        [0, 0, 0, 0]]
            if self.orientation == 3:
                return [[1, 0, 0, 0],
                        [1, 1, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 0, 0]]
        if self.shape == 5:  # Z piece
            if self.orientation == 0:
                return [[0, 0, 0, 0],
                        [1, 1, 0, 0],
                        [0, 1, 1, 0],
                        [0, 0, 0, 0]]
            if self.orientation == 1:
                return [[0, 0, 1, 0],
                        [0, 1, 1, 0],
                        [0, 1, 0, 0],
                        [0, 0, 0, 0]]
            if self.orientation == 2:
                return [[0, 0, 0, 0],
                        [1, 1, 0, 0],
                        [0, 1, 1, 0],
                        [0, 0, 0, 0]]
            if self.orientation == 3:
                return [[0, 0, 1, 0],
                        [0, 1, 1, 0],
                        [0, 1, 0, 0],
                        [0, 0, 0, 0]]
        if self.shape == 6:  # T piece
            if self.orientation == 0:
                return [[0, 0, 0, 0],
                        [1, 1, 1, 0],
                        [0, 1, 0, 0],
                        [0, 0, 0, 0]]
            if self.orientation == 1:
                return [[0, 1, 0, 0],
                        [1, 1, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 0, 0]]
            if self.orientation == 2:
                return [[0, 0, 0, 0],
                        [0, 1, 0, 0],
                        [1, 1, 1, 0],
                        [0, 0, 0, 0]]
            if self.orientation == 3:
                return [[0, 1, 0, 0],
                        [0, 1, 1, 0],
                        [0, 1, 0, 0],
                        [0, 0, 0, 0]]

    def get_color(self):
        if self.shape == 0:  # I piece
            return (0, 202, 229)
        if self.shape == 1:  # J piece
            return (255, 156, 0)
        if self.shape == 2:  # L piece
            return (3, 65, 174)
        if self.shape == 3:  # square shape
            return (210, 190, 0)
        if self.shape == 4:  # S piece
            return (0, 182, 65)
        if self.shape == 5:  # Z piece
            return (210, 0, 0)
        if self.shape == 6:  # T piece
            return (182, 0, 220)

    def get_next_piece_information_as_list(self):
        results = [0] * 7
        results[self.shape] = 1

        return results

    def get_origin(self):
        return [19, 4]

    def fast_drop(self, board):
        collision = False
        while not collision:
            self.origin[0] += 1
            if board.check_for_collision(self):
                self.origin[0] -= 1
                collision = True
        return self

    def drop(self, board):
        self.origin[0] += 1

    def undrop(self, board):
        self.origin[0] -= 1

    def move_left(self, board):
        self.origin[1] -= 1
        if board.check_for_collision(self):
            self.origin[1] += 1

    def move_right(self, board):
        self.origin[1] += 1
        if board.check_for_collision(self):
            self.origin[1] -= 1


class Evolution():
#    Evolution(setting, generation_size, neural_net_shape, operations, runs_per_nn, prob_swap, prob_mutation)
    def __init__(self, setting, gen_size, nn_shape, operations, runs_per_nn=1, prob_swap=0.2, prob_mutation=0.2):
        if setting == 'cpu':
            self.active = True
        else:
            self.active = False
        
        self.gen_size = gen_size
        self.nn_shape = nn_shape
# board_dimensions, steps, shape):
        self.prev_gen = []
        self.curr_gen = [NeuralNet(self.nn_shape, operations, nn_shape) for i in range(self.gen_size)]
        self.gen_count = 0

        self.curr_nn = 0
        self.runs_per_nn = runs_per_nn
        self.runs_on_curr_nn = 0

        self.prob_swap = prob_swap
        self.prob_mutation = prob_mutation

    def decide(self, board_state, next_pieces_state):
        return self.curr_gen[self.curr_nn].decide(board_state, next_pieces_state)

    def update_score(self, lines_cleared):
        self.curr_gen[self.curr_nn].update_score(lines_cleared)

    def save_metrics(self):
        self.curr_gen[self.curr_nn].save_metrics()

    def create_score(self):
        self.curr_gen[self.curr_nn].create_score()

    def create_next_generation(self):
        self.prev_gen = self.curr_gen
        self.curr_gen = []
        self.gen_count += 1

        self.prev_gen.sort(key=lambda x: x.avg_score, reverse=True)
        self.prev_gen[0].save_to_file('Results\\Gen ' + str(self.gen_count - 1) + '.txt', self.gen_count - 1, perform_save = False)

        scores = [nn.avg_score for nn in self.prev_gen]
        #expo_scores = [exp(score) for score in scores]
        #softmax_scores = [score / sum(expo_scores) for score in expo_scores]
        
        softmax_scores = [score / sum(scores) for score in scores]
        
        
        cumulative_scores = []
        for i in range(len(softmax_scores)):
            cumulative_scores.append(sum(softmax_scores[0:i]))

        print(scores[0:10])

        for i in range(self.gen_size):
            decision_var_1, decision_var_2 = random(), random()
            index_1, index_2 = 0, 0
            while decision_var_1 >= cumulative_scores[index_1] and index_1 < len(cumulative_scores) - 1: index_1 += 1
            while decision_var_2 >= cumulative_scores[index_2] and index_2 < len(cumulative_scores) - 1: index_2 += 1
            self.curr_gen.append(NeuralNet.create_cross_mutation(self.prev_gen[index_1], self.prev_gen[index_2],
                                                                 self.prob_swap).create_mutations(self.prob_mutation,
                                                                                                  sigma=0.05))

    def next_run(self):
        self.runs_on_curr_nn += 1
        if self.runs_on_curr_nn == self.runs_per_nn:
            self.create_score()
            self.curr_nn += 1
            self.runs_on_curr_nn = 0

        if self.curr_nn == self.gen_size:
            self.curr_nn = 0
            self.create_next_generation()

    def update_rng(self, val):
        self.curr_gen[self.curr_nn].update_rng(val)

    def update_history(self, val):
        self.curr_gen[self.curr_nn].update_history(val)
        

class NeuralNet():
    def __init__(self, board_dimensions, operations, shape):
        self.shape = shape
        self.layers = len(self.shape)
        self.operations = operations

        self.rng = []
        self.curr_rng = []
        self.history = []
        self.curr_history = []
        
        self.scores = []
        self.curr_score = 0
        self.avg_score = 0

        self.weights = []
        self.biases = []
        self.kernels = []
        self.kernel_biases = []
        
        for i in range(self.operations.count('convolution')):
            self.kernels.append(np.random.randn(3, 3))
            self.kernel_biases.append(random())
        
        for i in range(self.layers - 1):
            if type(self.shape[i]) == int:
                self.weights.append(2 * np.random.randn(self.shape[i + 1], self.shape[i]))
                self.biases.append(2 * np.random.randn(self.shape[i + 1], 1))
        
        #for i in range(self.layers - 1):
        #    self.weights.append(2 * np.random.randn(self.shape[i + 1], self.shape[i]))
        #    self.biases.append(2 * np.random.randn(self.shape[i + 1], 1))

    def update_rng(self, piece):
        self.curr_rng.append(piece.shape)

    def update_history(self, val):
        self.curr_history.append(val)


    def decide(self, board_state, next_pieces_state):
        convolution_step = -1
        kernel_biases_index = -1
        
        curr_state = board_state
        
        convolutions_section = self.shape[1:2 + len(self.operations)]
        for operation, dimensions in zip(self.operations, convolutions_section):
            results_matrix = np.empty(dimensions)
            if operation == 'convolution':
                convolution_step += 1
                kernel_biases_index += 1
                
                for i in range(dimensions[0]):
                    for j in range(dimensions[1]):
                        results_matrix[i][j] = np.sum(self.kernels[convolution_step] * curr_state[i:i+3, j:j+3]) + self.kernel_biases[kernel_biases_index]
                
                results_matrix[results_matrix < 0] = 0 # ReLU mapping
            
            if operation == 'pooling':
                for i in range(1, dimensions[0] - 1):
                    for j in range(1, dimensions[1] - 1):
                        results_matrix[i][j] = np.max(curr_state[2 * i:2 * (i + 1), 2 * j:2 * (i + j)])
            
            curr_state = results_matrix
        
        curr_state = curr_state.flatten()
        
        X = np.ndarray((len(curr_state) + len(next_pieces_state), 1))
        for i, val in enumerate(curr_state):
            X[i] = val
        for i, val in enumerate(next_pieces_state):
            X[i + len(curr_state)] = val
        
        Z = np.dot(self.weights[0], X) + self.biases[0]
        for i in range(1, len(self.weights)):
            A = np.tanh(Z)
            Z = np.dot(self.weights[i], A) + self.biases[i]

        A = 1 / (1 + np.exp(-Z))
        decision = self.get_softmax(A)

        return decision

    def get_softmax(self, A):
        A = A.tolist()
        A = [row[0] for row in A]

        exponential_A = [exp(x) for x in A]
        denominator_A = [x / sum(exponential_A) for x in exponential_A]

        return (denominator_A.index(max(denominator_A)))

    def update_score(self, lines_cleared):
        #score_options = [0, 100, 200, 300, 800]
        score_options = [0, 1, 2, 3, 8]
        self.curr_score += score_options[lines_cleared]

    def save_metrics(self):
        self.scores.append(self.curr_score)
        self.curr_score = 0

        self.rng.append(self.curr_rng)
        self.curr_rng = []
        self.history.append(self.curr_history)
        self.curr_history = []

    def create_score(self):
        self.avg_score = sum(self.scores) / len(self.scores)
        if self.avg_score == 0:
            self.avg_score = 0.001
        self.avg_score = 2 ** self.avg_score

    @classmethod
    def create_cross_mutation(cls, nn_1, nn_2, prob_swap):
        combined_nn = deepcopy(nn_1)

        for i, w in enumerate(nn_2.weights):
            if random() <= prob_swap:
                j = randint(0, len(w) - 1)
                combined_nn.weights[i][j:] = w[j:]

        for i, b in enumerate(nn_2.biases):
            if random() <= prob_swap:
                j = randint(0, len(b - 1))
                combined_nn.biases[i][j:] = b[j:]

        return combined_nn

    def create_mutations(self, prob_mutation, sigma=0.1):
        mutated_nn = deepcopy(self)

        for i, weights in enumerate(mutated_nn.weights):
            bool_vals = np.random.choice([True, False], weights.shape, p=[prob_mutation, 1 - prob_mutation])
            random_vals = np.random.randn(weights.shape[0], weights.shape[1])
            weights[bool_vals] += random_vals[bool_vals]
            mutated_nn.weights[i] = weights

        for i, biases in enumerate(mutated_nn.biases):
            bool_vals = np.random.choice([True, False], biases.shape, p=[prob_mutation, 1 - prob_mutation])
            random_vals = sigma * np.random.randn(biases.shape[0], biases.shape[1])
            biases[bool_vals] += random_vals[bool_vals]
            mutated_nn.biases[i] = biases
        
        for i, kernel in enumerate(mutated_nn.kernels):
            bool_vals = np.random.choice([True, False], kernel.shape, p=[prob_mutation, 1 - prob_mutation])
            random_vals = sigma * np.random.randn(kernel.shape[0], kernel.shape[1])
            kernel[bool_vals] += random_vals[bool_vals]
            mutated_nn.kernels[i] = kernel
        
        for i, kernel_bias in enumerate(mutated_nn.kernel_biases):
            if random() < prob_mutation:
                mutated_nn.kernel_biases[i] += random() - 0.5
        
        return mutated_nn
    
    @classmethod
    def generate_shape(cls, operations, board_shape, next_piece_count, feedforward_shape):
        (board_rows, board_cols) = board_shape
        shape = []
        shape.append([board_rows, board_cols])
        
        for operation in operations:
            if operation == 'convolution':
                board_rows -= 2
                board_cols -= 2
                shape.append([board_rows, board_cols])
            if operation == 'pooling':
                board_rows = ceil(board_rows / 2)
                board_cols = ceil(board_cols / 2)
                shape.append([board_rows, board_cols])
            
        shape.append(board_rows * board_cols + (7 * next_piece_count))
        shape = shape + feedforward_shape
        
        return shape
    
    def save_to_file(self, path, generation, perform_save=True):
        if not perform_save:
            return
        with open(path, 'a') as f:
            f.write('shape:' + ' '.join([str(val) for val in self.shape]) + '\n')
            f.write('next piece count: ' + str(int((self.shape[0] - 200) / 7)) + '\n')
            f.write('generation:' + str(generation) + '\n')

            for i, (score, history, rng) in enumerate(zip(self.scores, self.history, self.rng)):
                f.write('score run ' + str(i) + ':' + str(score) + '\n')
                f.write('history run ' + str(i) + ':' + ' '.join([str(val) for val in history]) + '\n')
                f.write('rng run ' + str(i) + ': ' + ' '.join([str(val) for val in rng]) + '\n')

            for i, weight in enumerate(self.weights):
                f.write('weights section ' + str(i) + ' shape=' + str(weight.shape[0]) + ' ' + str(weight.shape[1]))
                for j, row in enumerate(weight):
                    f.write('\nweights section ' + str(i) + ' subsection ' + str(j) + '=')
                    for val in row:
                        f.write(str(val))
                        f.write(' ')
            f.write('\n')
            for i, bias in enumerate(self.biases):
                bias_content = [str(val) for val in np.nditer(bias)]
                f.write('biases section ' + str(i) + ' =' + ' '.join(bias_content) + '\n')


class Replay():
    def __init__(self, setting, directory):
        if setting == 'replay':
            self.active = True
        else:
            self.active = False
            return None
        txt_paths = []
        for dirpath, dirs, files in os.walk(directory):
            for filename in files:
                fname = os.path.join(dirpath, filename)
                if fname.endswith('.txt'):
                    txt_paths.append(fname)

        txt_paths.sort(key=lambda x: int(re.findall(r'\d+', x)[-1]))

        self.rngs = []
        self.decisions = []

        self.curr_run = 0
        self.curr_rng_step = 0
        self.curr_decision_step = 0

        for txt_path in txt_paths:
            with open(txt_path, 'r') as f:
                lines = [line for line in f]

            tags = ['shape', 'next piece count', 'genertion', 'score', 'history', 'rng', 'weights', 'biases']
            tag_indexes = [[] for tag in tags]

            for i, line in enumerate(lines):
                for j, tag in enumerate(tags):
                    if tag in line:
                        tag_indexes[j].append(i)

            for history_index in tag_indexes[tags.index('history')]:
                history = lines[history_index]
                history = history.split(':')[1]
                history = history.strip().split(' ')
                history = [int(h) for h in history]

                self.decisions.append(history)

            for rng_index in tag_indexes[tags.index('rng')]:
                rng = lines[rng_index]
                rng = rng.split(':')[1]
                rng = rng.strip().split(' ')
                rng = [int(r) for r in rng]

                self.rngs.append(rng)

        self.total_runs = len(self.rngs)

    def get_next_rng(self):
        next_rng = self.rngs[self.curr_run][self.curr_rng_step]
        self.curr_rng_step += 1

        return next_rng

    def get_next_decision(self):
        # print(self.curr_decision_step)
        next_decision = self.decisions[self.curr_run][self.curr_decision_step]

        self.curr_decision_step += 1
        if self.curr_decision_step == len(self.decisions[self.curr_run]):
            self.curr_run = (self.curr_run + 1) % self.total_runs
            self.curr_rng_step = 0
            self.curr_decision_step = 0
            print('reset2!', self.curr_run)

        return next_decision


class Game():
    def __init__(self, setting, generation_size, operations, feedforward_shape, runs_per_nn=1, prob_swap=0.2, prob_mutation=0.2, replay_directory=None, next_piece_count=3):
        self.tile_length = 20
        self.rows = 40
        self.hidden_rows = 20
        self.columns = 10
        self.right_side_space = 6
        self.next_piece_count = next_piece_count
        self.render = True
        self.BACKGROUND = (0, 0, 0)
        self.fps_speeds = [3, 10, 30, 60, 100000]
        self.fps_index = len(self.fps_speeds) - 1
        self.FPS = self.fps_speeds[self.fps_index]
        self.fpsClock = pygame.time.Clock()
        self.WINDOW_WIDTH = (self.columns + self.right_side_space) * self.tile_length + 1
        self.WINDOW_HEIGHT = (self.rows - self.hidden_rows) * self.tile_length + 1
        self.WINDOW = pygame.display.set_mode((self.WINDOW_WIDTH, self.WINDOW_HEIGHT))
        pygame.display.set_caption('Tetris')
        
        self.board = Board(self.rows, self.hidden_rows, self.columns, self.tile_length)
        self.replay_info = Replay(setting, replay_directory)
        
        neural_net_shape = NeuralNet.generate_shape(operations, (self.rows - self.hidden_rows, self.columns), next_piece_count, feedforward_shape)
        print(neural_net_shape)
        
        self.evolution = Evolution(setting, generation_size, neural_net_shape, operations, runs_per_nn, prob_swap, prob_mutation)
        
        self.pieces = self.initialize_first_pieces(self.evolution, self.replay_info, self.next_piece_count)

        while True:
            self.run(setting)

        pygame.quit()
        sys.exit()

    def run(self, setting):
        decision = self.get_input(setting, self.pieces)
        self.input_decision(decision, self.pieces[0], self.board)  # move the piece based on user/cpu/replay input

        piece_stays_alive = self.board.check_if_piece_alive(self.pieces[0])
        if piece_stays_alive:
            self.pieces[0].drop(self.board)
        else:
            self.board.add_piece(self.pieces[0])
            self.get_next_piece(self.pieces, self.evolution, self.replay_info)

            lines_cleared = self.board.clear_lines()
            self.evolution.update_score(lines_cleared)

        if not piece_stays_alive and self.board.check_for_gameover(self.pieces[0]):
            # print('game over!')
            if setting == 'replay':
                if self.replay_info.curr_decision_step != 0:
                    print('replay did not roll over to 0 on its own')
                    print(self.replay_info.curr_decision_step, len(self.replay_info.decisions[self.replay_info.curr_run]))
                    self.replay_info.curr_rng_step = 0
                    self.replay_info.curr_decision_step = 0
            if setting == 'cpu':
                self.evolution.save_metrics()
                self.evolution.next_run()

            self.board.clear()
            
            self.pieces = self.initialize_first_pieces(self.evolution, self.replay_info, self.next_piece_count)

        self.draw_game(self.render, piece_stays_alive)

        pygame.display.update()
        self.fpsClock.tick(self.FPS)

    def draw_game(self, render, piece_stays_alive):
        if not render:
            return

        self.WINDOW.fill(self.BACKGROUND)
        for a, next_piece in enumerate(self.pieces[1:]):
            for i in range(4):
                for j in range(4):
                    if next_piece.positions[i][j]:
                        pygame.draw.rect(self.WINDOW, next_piece.color, [(j + self.columns + 1) * self.tile_length,
                                                                         (i + 1 + (4 * a)) * self.tile_length,
                                                                         self.tile_length, self.tile_length])

        if piece_stays_alive:
            self.board.add_moving_piece(self.pieces[0])

        self.board.draw(self.WINDOW)
        pygame.display.update()

        if piece_stays_alive:
            self.board.remove_current_piece()
    
    def initialize_first_pieces(self, evolution, replay_info, next_piece_count):
        return [Piece(evolution, replay_info) for i in range(1 + next_piece_count)]
        
    
    
    def get_next_piece(self, pieces, evolution, replay_info):
        pieces.append(Piece(evolution, replay_info))
        pieces.pop(0)
    
    
    def get_user_input(self):
        game_action_decision = 0

        for event in pygame.event.get():
            if event.type == pygame.locals.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    self.fps_index = max(0, self.fps_index - 1)
                    self.FPS = self.fps_speeds[self.fps_index]
                if event.button == 2:
                    self.render = not self.render
                if event.button == 3:
                    self.fps_index = min(len(self.fps_speeds) - 1, self.fps_index + 1)
                    self.FPS = self.fps_speeds[self.fps_index]

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    game_action_decision = 1
                if event.key == pygame.K_RIGHT:
                    game_action_decision = 2
                if event.key == pygame.K_r:  # rotate clockwise
                    game_action_decision = 3
                if event.key == pygame.K_e:  # rotate counterclockwise
                    game_action_decision = 4
                if event.key == pygame.K_SPACE:  # drop the piece
                    game_action_decision = 5

        return game_action_decision

    def get_cpu_input(self, pieces):
        self.board.get_board_information_as_2d_array(pieces[0])
        board_state = self.board.get_board_information_as_2d_array(pieces[0])
        next_pieces_state = []
        for next_piece in pieces[1:]:
            next_pieces_state += next_piece.get_next_piece_information_as_list()

        return self.evolution.decide(board_state, next_pieces_state)

    def get_input(self, setting, pieces):
        human_decision = self.get_user_input()  # also gets inputs like if window is closed, clicks
        if setting == 'cpu':
            decision = self.get_cpu_input(pieces)
            self.evolution.update_history(decision)

        elif setting == 'human':
            decision = human_decision

        elif setting == 'replay':
            decision = self.replay_info.get_next_decision()

        return decision

    def input_decision(self, decision, piece, board):
        if decision == 0:
            pass
        if decision == 1:
            piece.move_left(board)
        if decision == 2:
            piece.move_right(board)
        if decision == 3:
            piece.rotate_clockwise(board)
        if decision == 4:
            piece.rotate_counterclockwise(board)
        if decision == 5:
            piece.fast_drop(board)


if __name__ == '__main__':
    setting_options = ['cpu', 'human', 'replay']
    setting_options_index = 0
    
    operations = ['convolution'] #'convolution', 'pooling'

    game = Game(setting_options[setting_options_index], generation_size=1000, operations=operations, feedforward_shape=[50, 20, 6], runs_per_nn=1, prob_swap=0.1, prob_mutation=0.05, replay_directory='Results',
                next_piece_count=1)
    game.run(setting_options[setting_options_index])

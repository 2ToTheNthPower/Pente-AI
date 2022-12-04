from manim import *
import numpy as np
import networkx as nx
import random
import manim_ml
from scipy.ndimage.interpolation import rotate

from manim_ml.neural_network.layers import EmbeddingLayer
from manim_ml.neural_network.layers import Convolutional3DLayer
from manim_ml.neural_network.layers import FeedForwardLayer
from manim_ml.neural_network.layers import ImageLayer
from manim_ml.neural_network.neural_network import NeuralNetwork

# This code builds the following series of scenes:
#
#   1. Explanation of Game Board and Win Conditions
#   2. Explanation of Graph representation of possible game states.
#   3. Explanation of Monte Carlo Tree Search.
#   4. Explanation of Neural Network in MCTS


class SceneOne(MovingCameraScene):
    def construct(self):
        
        # self.add_sound("CelloSuite.wav", gain=-20)
        self.wait(2)

        # PART 1: Explaining how the game works
        
        # part1 = Text("Part 1: The Data")
        # self.play(Write(part1))
        # self.wait()
        # self.play(FadeOut(part1))

        # self.wait()

        def drawBoard(run_time = .2, length = 6, board_size = 19, x = 0, y = 0):

            square = Square(side_length=length)
            square.set_fill(BLACK, opacity=0.5)
            square.move_to(np.array([x, y, 0]))
            self.play(Create(square), run_time=run_time)

            g = VGroup()
            g.add(square)

            horizontal_lines = []
            vertical_lines = []

            for i in range(board_size):
                hl = Line(color=WHITE)
                hl.put_start_and_end_on(start=np.array([x - 3., y + (9 - i) * (length/2) / (9), 0.]), end=np.array([x + 3., y + (9 - i) * (length/2) / (9), 0.]))

                horizontal_lines.append(hl)

                vl = Line(color=WHITE)
                vl.put_start_and_end_on(start=np.array([x + (9 - i) * (length/2) / (9), y - 3., 0.]), end=np.array([x + (9 - i) * (length/2) / (9), y + 3., 0.]))

                vertical_lines.append(vl)

                g.add(hl)
                g.add(vl)

                self.play(Create(hl, run_time=run_time), Create(vl, run_time=run_time))

            return g
        
        g1 = drawBoard(run_time=.2)

        self.wait(6)

        # Zoom to show how to win
        self.camera.frame.save_state()
        self.play(self.camera.frame.animate.set(width=3))
        self.wait()

        pieces = []

        for i in range(5):
            circle = Circle(radius=.1, color=BLUE)
            circle.set_fill(BLUE, opacity=1)
            circle.move_to((i - 2) * 3/9)
            pieces.append(circle)
            self.play(Create(circle, run_time=.2))

        self.wait()
        
        for i in range(len(pieces)):
            if i != 2:
                self.play(pieces[i].animate.match_y(pieces[2]), run_time=.2)

        self.wait()

        for i in range(len(pieces)):
            if i != 2:
                self.play(pieces[i].animate.move_to(
                    np.array([(i - 2) * 3/9., - (i - 2) * 3/9., 0.])), 
                    run_time=.2)

        self.wait()

        for i in range(len(pieces)):
            if i != 2:
                self.play(pieces[i].animate.match_x(pieces[2]), run_time=.2)

        self.wait()
        
        animations = []
        for i in range(len(pieces)):
            animations.append(FadeOut(pieces[i], scale=0.5))

        self.play(AnimationGroup(*animations, lag_ratio=0))

        self.wait()

        self.play(self.camera.frame.animate.move_to(np.array([1/6, 1/6, 0.])))

        self.wait()

        pieces = []

        circle0 = Circle(radius=.1, color=BLUE)
        circle0.set_fill(BLUE, opacity=1)

        circle1 = Circle(radius=.1, color=BLUE)
        circle1.set_fill(BLUE, opacity=1)
        circle1.move_to(np.array([1/3, 1/3, 0]))

        circle2 = Circle(radius=.1, color=RED)
        circle2.set_fill(RED, opacity=1)
        circle2.move_to(np.array([-1/3, -1/3, 0]))

        circle3 = Circle(radius=.1, color=RED)
        circle3.set_fill(RED, opacity=1)
        circle3.move_to(np.array([2/3, 2/3, 0]))


        self.play(Create(circle0))
        self.play(Create(circle3))
        self.play(Create(circle1))
        self.play(Create(circle2))

        self.wait()

        # self.play()
        # self.play()

        # self.camera.frame.save_state()
        # self.play()
        # self.wait()

        captured = Text("captured")
        captured.next_to(np.array([3, 0, 0]), RIGHT, buff=1)

        red = Text("Red", color=RED)
        # circlered.set_fill(RED, opacity=1)
        red.next_to(captured, UP)
        # self.play(Write(red))

        blue = Text("1 Pair", color=BLUE)
        blue.next_to(captured, DOWN)

        # circleblue = Circle(radius=.25, color=BLUE)
        # circleblue.set_fill(BLUE, opacity=1)
        # circleblue.next_to(captured, DOWN)
        # circleblue2 = circleblue.copy()

        # circleblue2.move_to(np.array([circleblue2.get_x() - .35, circleblue2.get_y(), 0]))
        # circleblue.move_to(np.array([circleblue.get_x() + .35, circleblue.get_y(), 0]))


        self.play(Restore(self.camera.frame),
                    Write(red), 
                    Write(captured),
                    Write(blue),
                    circle0.animate.move_to(np.array([10, 0, 0])), 
                    circle1.animate.move_to(np.array([10, 0, 0])),
                    FadeOut(circle0),
                    FadeOut(circle1)
        )

        self.play(self.camera.frame.animate.move_to(np.array([2, 0, 0])))

        self.wait()

        self.play(self.camera.frame.animate.move_to(np.array([0, 0, 0])), 
                    FadeOut(red), 
                    FadeOut(captured), 
                    FadeOut(blue),
                    FadeOut(circle2),
                    FadeOut(circle3))

        self.wait()

        # self.play(FadeOut(*g1))

        self.play(self.camera.frame.animate.set(width=40).move_to(np.array([0, -3, 0])))

        for i in range(-16, 17, 8):
            g2 = drawBoard(run_time=0.01, x=i, y=-8)

        self.wait()

        # for i in range


        for i, j, x, y in zip([(0, -3), (0, -3), (0, -3), (0, -3), (0, -3)], [(-16, -5), (-8, -5), (0, -5), (8, -5), (16, -5)], [10, 3, 5, 9, 17], [10, 12, 15, 8, 18]):

            arrow1 = Arrow(np.array([i[0], i[1], 0]), np.array([j[0], j[1], 0]), buff=0)
            self.play(Create(arrow1))

            if j[0] != 0:
                pos = UP
                buff = 0
            else:
                pos = RIGHT
                buff = 1

            # move1 = Text(f"@ ({x}, {y})")
            # move1.next_to(arrow1, pos, buff=buff)

            # circle0 = Circle(radius=.3)
            # circle0.set_fill(RED, opacity=1)
            # circle0.next_to(move1, LEFT)

            circle1 = Circle(radius=.2)
            circle1.set_fill(RED, opacity=1)
            circle1.move_to(np.array([j[0] + (x - 1) / 3 - 3, j[1] + (y - 1) / 3 - 6, 0]))
            
            # Add graph edges
            # self.play(Write(move1), Create(circle0))
            self.play(Create(circle1))

        self.wait()

        # self.clear()

        self.play(self.camera.frame.animate.move_to(np.array([0, -10, 0])))

        offset = -8

        for i in range(-16, 17, 8):
            g2 = drawBoard(run_time=0.005, x=i, y=-8 + offset)

        # self.wait()

        # for i in range


        for i, j in zip([(0, -3 + offset), (0, -3 + offset), (0, -3 + offset), (0, -3 + offset), (0, -3 + offset)], [(-16, -5 + offset), (-8, -5 + offset), (0, -5 + offset), (8, -5 + offset), (16, -5 + offset)]):

            arrow1 = Arrow(np.array([i[0], i[1], 0]), np.array([j[0], j[1], 0]), buff=0)
            self.play(Create(arrow1))

            circle1 = Circle(color=RED, radius=.2)
            circle1.set_fill(RED, opacity=1)
            circle1.move_to(np.array([j[0] + (5 - 1) / 3 - 3, j[1] + (15 - 1) / 3 - 6, 0]))

            circle2 = Circle(color=BLUE, radius=.2)
            circle2.set_fill(BLUE, opacity=1)

            x = random.randrange(1,19)
            y =  random.randrange(1,19)

            circle2.move_to(np.array([j[0] + (x - 1) / 3 - 3, j[1] + (y - 1) / 3 - 6, 0]))
            
            self.play(Create(circle1), Create(circle2))

        self.play(self.camera.frame.animate.set(width=15).move_to(np.array([16, -16, 0])))
        self.wait()

        print(x, y)

        zeros = np.zeros(shape=(19,19))
        zeros[x-1,y-1] = -1
        zeros[5-1,15-1] = 1
        board = zeros.astype(int)
        board = rotate(board, angle=90)
        board = board.astype(int)
        

        self.clear()

        m = Matrix(board, v_buff=.5, h_buff=.65).scale(.675)
        m.move_to(np.array([16, -16, 0]))
        self.play(Create(m))

        self.wait()


class SceneTwo(MovingCameraScene):
    def construct(self):
        LAYOUT_CONFIG = {"vertex_spacing": (0.25, .25)}
        G = nx.random_tree(250, seed=100)
        graph = Graph(list(G.nodes), list(G.edges), layout="tree", root_vertex=0, layout_config=LAYOUT_CONFIG)

        self.play(Create(graph))
        
        self.camera.frame.save_state()
        self.play(self.camera.frame.animate.set(width=20))

        self.wait(12)

        self.play(self.camera.frame.animate.move_to(np.array([5, 0, 0])))

        # self.play(self.camera.frame.animate.set(width=3))

        outcome = Tex(r"$winrate = \frac{\sum{Outcomes}}{Visit Count}$", font_size=80)
        outcome.next_to(graph, RIGHT, buff=.5)
        self.play(Write(outcome))

        rect1 = Rectangle(width=2.0, height=3.0, color="red")
        rect1.move_to(np.array([4.25, 0.75, 0]))

        self.wait(14)
        self.play(Create(rect1))

        self.play(self.camera.frame.animate.set(height = rect1.height * 1.5).move_to(np.array([rect1.get_x() + 2, rect1.get_y(), 0])), 
                    FadeOut(outcome),
                    FadeOut(rect1))

        win = Text("1", font_size=20)
        win.next_to(np.array([5, 0.5, 0]), RIGHT, buff=0.1)
        self.play(Create(Circle(color=RED, radius=.075).set_fill(RED, opacity=1).move_to(np.array([4.9, 0.5, 0]))))

        loss = Text("0", font_size=20)
        loss.next_to(np.array([4.875, 0.25, 0]), RIGHT, buff=0.1)
        self.play(Create(Circle(color=BLUE, radius=.075).set_fill(BLUE, opacity=1).move_to(np.array([4.775, 0.25, 0]))))

        draw = Tex(r"$\frac{1}{2}$", font_size=20)
        draw.next_to(np.array([4.375, -.5, 0]), RIGHT, buff=0.1)
        self.play(Create(Circle(color=GREY, radius=.075).set_fill(GREY, opacity=1).move_to(np.array([4.275, -.5, 0]))))


        self.play(Write(win), Write(loss), Write(draw))

        # Fill out rest of subsection of graph

        self.play(Create(Circle(color=BLUE, radius=.075).set_fill(BLUE, opacity=1).move_to(np.array([4.525, 0.25, 0]))))
        self.play(Create(Circle(color=RED, radius=.075).set_fill(RED, opacity=1).move_to(np.array([4.025, -.25, 0]))))
        self.play(Create(Circle(color=RED, radius=.075).set_fill(RED, opacity=1).move_to(np.array([3.9, 0.5, 0]))))
        self.play(Create(Circle(color=BLUE, radius=.075).set_fill(BLUE, opacity=1).move_to(np.array([3.65, 0.5, 0]))))
        self.play(Create(Circle(color=RED, radius=.075).set_fill(RED, opacity=1).move_to(np.array([3.4, 0.75, 0]))))
        self.play(Create(Circle(color=RED, radius=.075).set_fill(RED, opacity=1).move_to(np.array([4.14, 1.25, 0]))))

        self.wait(7)

        # Sum all effects on top node
        arrow = Arrow(end=np.array([4.35, 2, 0]), start=np.array([7, 2, 0]))
        self.play(Create(arrow))

        n = Tex(r"$N_i = 9$", font_size=30)
        n.next_to(arrow, RIGHT)

        w = Tex(r"$W_i = 5.5$", font_size=30)
        w.next_to(n, DOWN)

        average = Tex(r"$\frac{W_i}{N_i} = 0.61$", font_size=30)
        average.next_to(w, DOWN)

        self.play(Write(n))
        self.play(Write(w))
        self.play(Write(average))

        self.wait(23)

        self.play(FadeOut(arrow), 
                    FadeOut(n), 
                    FadeOut(w), 
                    FadeOut(average))

        self.play(Restore(self.camera.frame))

        self.wait()

        self.clear()

        self.wait(2)

        atoms = Tex(r"$10^{80}$")

        game_states = Tex(r"$10^{170}$")

        self.play(Write(atoms))

        self.wait(17)

        self.play(Transform(atoms, game_states))

class SceneThree(MovingCameraScene):
    def construct(self):
        LAYOUT_CONFIG = {"vertex_spacing": (0.25, .5)}
        G = nx.Graph()
        G.add_nodes_from([0])

        graph = Graph(list(G.nodes), list(G.edges), layout="tree", root_vertex=0, layout_config=LAYOUT_CONFIG)

        self.play(Create(graph))

        self.wait(11)

        nodes_per_layer = 5
        layers = 8

        prev_nodes = [0] * nodes_per_layer
        remove_nodes = []

        for i in range(1, layers * nodes_per_layer, nodes_per_layer):
            new_nodes = list(range(i, i + nodes_per_layer))
            G.add_nodes_from(new_nodes)
            new_edges = list(zip(prev_nodes, new_nodes))
            G.add_edges_from(new_edges)
            
            child = random.choice(new_nodes)
            prev_nodes = [child] * nodes_per_layer
    
            graph.add_vertices(*new_nodes, positions={k:np.array([random.uniform(0,0.01),random.uniform(0,0.01),0]) for k in new_nodes})
            graph.add_edges(*new_edges)
          
            self.play(graph.animate.change_layout(layout="tree", root_vertex=0))

            self.wait()
            
            new_nodes.remove(child)

            graph.remove_vertices(*new_nodes)


        # for i in range(1, layers * nodes_per_layer, nodes_per_layer):
        #     new_nodes = list(range(i, i + nodes_per_layer))
        #     G.add_nodes_from(new_nodes)
        #     new_edges = list(zip(prev_nodes, new_nodes))
        #     G.add_edges_from(new_edges)
            
        #     child = random.choice(new_nodes)
        #     prev_nodes = [child] * nodes_per_layer
        #     graph2 = Graph(list(G.nodes), list(G.edges), layout="tree", root_vertex=0, layout_config=LAYOUT_CONFIG)
            

        #     if len(remove_nodes) > 0:
        #         graph2.remove_vertices(*remove_nodes)
            
        #     self.play(Transform(graph, graph2))

        #     if 0 not in prev_nodes:
        #         new_nodes.remove(child)
                
        #         # remove_edges = new_edges.remove(())
        #         graph2.remove_vertices(*new_nodes)
        #         # G.remove_nodes_from(new_nodes)
        #         remove_nodes += new_nodes

        #         self.play(Transform(graph, graph2))

        


        self.wait()

        self.play(FadeOut(graph))


class SceneFour(MovingCameraScene):
    def construct(self):

        upper_confidence_bound = Text("Upper Confidence Bound Score for Trees")

        self.play(Write(upper_confidence_bound))

        self.wait(4)

        UCB_score = MathTex(
            "UCB(s) =","\\frac{W_s}{N_s}","+",
            "c P(s)", "\\frac{\\sqrt{N_p}}{1 + N_s}"
        )

        self.play(Transform(upper_confidence_bound, UCB_score))

        self.wait(30)

        # framebox0 = SurroundingRectangle(UCB_score[0], buff = .1)
        framebox1 = SurroundingRectangle(UCB_score[1], buff =.1)
        framebox3 = SurroundingRectangle(UCB_score[3], buff =.1)
        framebox4 = SurroundingRectangle(UCB_score[4], buff =.1)

        self.play(
            Create(framebox1),
        )
        self.wait(7)

        self.play(
            ReplacementTransform(framebox1,framebox4),
        )

        self.wait(16)
        self.play(
            ReplacementTransform(framebox4,framebox3),
        )

        self.wait(10)

        self.play(FadeOut(framebox3))


        # # Illustrating what happens to the UCB score for unexplored and highly explored nodes.

        # UCB_score2 = MathTex(
        #     "UCB(s) \\approx", "\\infty"
        # )

        # self.play(Transform(upper_confidence_bound, UCB_score2))

        # self.wait()

        # UCB_score3 = MathTex(
        #     "UCB(s) \\approx", "\\frac{W_s}{N_s}"
        # )

        # self.play(Transform(upper_confidence_bound, UCB_score3))

        # self.wait()

class SceneFive(MovingCameraScene):
    def construct(self):

        # https://github.com/manimml/ManimML/blob/main/examples/cnn/cnn.py

        board = np.ones((19,19))
        board *= 255/2

        board[3,7] = 255
        board[5,6] = 0
        
        n = 50

        numpy_image = np.kron(board, np.ones((n,n)))
        probs = np.kron(np.random.uniform(size=(19,19)) * 255, np.ones((n,n)))
        # Make nn
        nn = NeuralNetwork([
            ImageLayer(numpy_image, height=3.5),
            Convolutional3DLayer(3, 3, 3, filter_spacing=0.2),
            Convolutional3DLayer(5, 2, 2, filter_spacing=0.2),
            Convolutional3DLayer(10, 2, 1, filter_spacing=0.2),
            FeedForwardLayer(5, rectangle_stroke_width=4, node_stroke_width=4).scale(2),
            FeedForwardLayer(5, rectangle_stroke_width=4, node_stroke_width=4).scale(2),
            Convolutional3DLayer(10, 2, 1, filter_spacing=0.2),
            Convolutional3DLayer(5, 2, 2, filter_spacing=0.2),
            Convolutional3DLayer(3, 3, 3, filter_spacing=0.2),
            ImageLayer(probs, height=3.5),
        ], layer_spacing=0.2)
        nn.scale(0.6)
        nn.move_to(ORIGIN)
        # nn.shift(UP*1.8)

        self.play(Create(nn))

class SceneSix(MovingCameraScene):
    def construct(self):

        # Let's put all the pieces together

        # We begin to build a graph

        LAYOUT_CONFIG = {"vertex_spacing": (0.25, .25)}

        G = nx.random_tree(200, seed=100)
        graph = Graph(list(G.nodes), list(G.edges), layout="tree", root_vertex=0, layout_config=LAYOUT_CONFIG)

        self.play(Create(graph))

        self.camera.frame.save_state()
        self.play(self.camera.frame.animate.move_to(np.array([20, 0, 0])))

        board = np.ones((19,19))
        board *= 255/2

        board[3,7] = 255
        board[5,6] = 0
        
        n = 50

        numpy_image = np.kron(board, np.ones((n,n)))
        probs = np.kron(np.random.uniform(size=(19,19)) * 255, np.ones((n,n)))
        # Make nn
        nn = NeuralNetwork([
            ImageLayer(numpy_image, height=3.5),
            Convolutional3DLayer(3, 3, 3, filter_spacing=0.2),
            Convolutional3DLayer(5, 2, 2, filter_spacing=0.2),
            Convolutional3DLayer(10, 2, 1, filter_spacing=0.2),
            FeedForwardLayer(5, rectangle_stroke_width=4, node_stroke_width=4).scale(2),
            FeedForwardLayer(5, rectangle_stroke_width=4, node_stroke_width=4).scale(2),
            Convolutional3DLayer(10, 2, 1, filter_spacing=0.2),
            Convolutional3DLayer(5, 2, 2, filter_spacing=0.2),
            Convolutional3DLayer(3, 3, 3, filter_spacing=0.2),
            ImageLayer(probs, height=3.5),
        ], layer_spacing=0.2)
        nn.scale(0.6)
        nn.move_to(np.array([20,0,0]))
        # nn.shift(UP*1.8)

        self.play(Create(nn), run_time=1)

        self.wait(3)

        self.play(self.camera.frame.animate.move_to(np.array([0, 0, 0])))

        self.wait()

        self.play(self.camera.frame.animate.move_to(np.array([20, 0, 0])))

        self.wait()

        self.play(self.camera.frame.animate.move_to(np.array([0, 0, 0])))

        self.wait()

        self.play(self.camera.frame.animate.move_to(np.array([20, 0, 0])))

        self.wait()

        self.play(self.camera.frame.animate.move_to(np.array([0, 0, 0])))

        self.wait()

        self.play(self.camera.frame.animate.move_to(np.array([20, 0, 0])))

        self.wait()

class SceneSeven(MovingCameraScene):
    def construct(self):
        # AlphaZero Results
        results = Text("AlphaZero Results for Chess, Shogi, and Go")
        image = ImageMobject("AlphaZeroElo.png")
        image.width = 12
        results.next_to(image, UP)
        self.play(Write(results))
        self.play(FadeIn(image))
        self.wait(15)

        # My results
        our_results = Text("Our Results for Pente")
        our_plot = ImageMobject("train_history.png")
        elo = Text("Elo Rating", font_size=20)
        elo.next_to(our_plot, LEFT)
        iterations = Text("Iteration", font_size=20)
        iterations.next_to(our_plot, DOWN)

        self.play(FadeOut(image))
        self.play(FadeIn(our_plot), Write(elo), Write(iterations))

        our_results.next_to(our_plot, UP)
        self.play(ReplacementTransform(results, our_results))

        self.wait(15)

class SceneEight(MovingCameraScene):
    def construct(self):
        time = Text("Waste of time?")
        energy = Text("Waste of energy?")
        money = Text("Waste of money?")

        time.next_to(energy, UP)
        money.next_to(energy, DOWN)

        self.play(Write(time))
        self.wait(3)
        self.play(Write(energy))
        self.wait(3)
        self.play(Write(money))
        self.wait(6)

        self.play(FadeOut(time), FadeOut(money), FadeOut(energy))

        date = Text("October 5, 2022")

        self.wait(4)
        self.play(Write(date))

        self.wait(4)
        self.play(FadeOut(date))

        image = ImageMobject("alphatensor.png")
        image.width = 12
        self.wait()
        self.play(FadeIn(image))
        self.wait(10)

        self.play(FadeOut(image))

        image2 = ImageMobject("alphatensor2.png")
        image2.width = 12

        self.play(FadeIn(image2))
        self.wait(10)
        self.play(FadeOut(image2))

class SceneNine(MovingCameraScene):
    def construct(self):

        thanks = Text("Thanks to:")
        deepmind = Text("DeepMind - For Cool Ideas", font_size=20)
        blue = Text("3Blue1Brown - For a Cool Animation Library", font_size=20)
        lud = Text("Lud and Schlatt's Musical Emporium - For Copyright Free Music", font_size=20)
        ami = Text("Dr. Ami Gates - For Excellent Teaching Work", font_size=20)

        deepmind.next_to(blue, UP)
        thanks.next_to(deepmind, UP)
        lud.next_to(blue, DOWN)
        ami.next_to(lud, DOWN)

        self.play(Write(thanks))
        self.wait()
        self.play(Write(deepmind))
        self.wait()
        self.play(Write(blue))
        self.wait()
        self.play(Write(lud))
        self.wait(3)

        self.play(Write(ami))

        self.wait(10)
        self.play(FadeOut(thanks), FadeOut(deepmind), FadeOut(blue), FadeOut(lud), FadeOut(ami))





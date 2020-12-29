An implementation of self-driving car based on Genetic Algorithm

## Repository

* **model**

  The model is a fully connected neural network which corresponds to an individual in genetic algorithm. NN has 4 hidden layers, an input layer of 35 units and an output layer of 3 units. Of 35 inputs, 30 of which are Lidar data and others are states including distance to the target, sign and value of direction angle of the target, etc. The outputs of NN predicts the velocity and steering angle of the next step of the car.

* **racetrack**

  Racetrack is used to generate maps from a random path consisting of a plurality of line segments end by end. Each segment has a certain limitation of length and angle.

* **simulation**

  Simulation is a class that simulate the car in racetrack, including measuring the distance to the wall and calculating position(x,y,theta) of next step from current position, velocity and steering angle.

* **environment**

  Environment is the key of the code that regulates the judging criteria of hitting the wall, checkpoints, goal and determines the fitness function.

* **genetic**

  Genetic is another key of the code: Firstly, define the regulation of crossover and mutation; Secondly, calculate reward of an individual according the fitness function; Thirdly, record the final value of rewards of one generation in descending order ;Fourthly, generate population of the next generation on the basis of crossover, mutation, elite; Lastly, call former defined functions to train the model.

  ## Implementation

  Curriculum learning is introduced to simplify the training procedure and save computational time, there are three bunch of model sets for three types of maps.

  In the first training round, function ’train’ in ‘genetic.py’ is called to initialize agents and train them in the straight lanes with fixed orientation.
  Then,  the trained models of previous step continue to be trained in ‘Train2’ in ‘genetic.py’ in updated maps. This process is repeated until the car is able to self-driving in maps with multiple turns.

  ‘Evaluation’ in ‘genetic_ev.py’ is called to evaluate the models in each case.

  The implementation of steps mentioned above is shown in the Jupter notebook ‘ note_main.ipynb’.

  ## Description of models get in different stages

  racecar_deqinracecar_50_fnn_straight : the model can drive on a straight track with fixed orientation
  racecar_deqinracecar_50_fnn_angle: the model can drive on a straight track with changeable orientation
  racecar_deqinracecar_50_fnn_turn2: the model can drive through two turns
  racecar_deqinracecar_50_fnn_turn7: the model can drive through maximum 7 turns
  racecar_deqinracecar_30_fnn_turn20_final: the model can drive through 20 turns
#this code is the workbench for q-learning
#it consists on a lifting particle that must reach a certain height
#it is only subjected to gravity
#Force applied to the particle might be fixed 9.9 or 9.7N

import numpy as np
import math
import random
import matplotlib.pyplot as plt
from models import*


#INITIALIZE VARIABLES
######################


#STATES (ANGLES) are discretized -0.35, -0.34, -0,33,... 0,34, 0,35 and speed is discretized in
n_pos=71
ANGLES=np.linspace(-0.35,0.35,71)

#SPEEDS are discretized -1 rad/s, -0.99, -0.98 ... 0.98, 0,99 1 rad/s.
n_speeds=200
ANGULAR_VEL=np.linspace(-1,1,n_speeds)

#ROWS=    States (71*200=14200 rows)
#COLUMNS= Actions (Left - Right)
Rows=n_pos*n_speeds
Columns=2
Actions=(-0.1, 0.1) #-pivot_x or +pivot_x
Final_angle=0.0
Final_omega=0.0

pivot_x=10.0 

#time steps
n_items=302
x=np.linspace(0,301,n_items) #time for obtaining the result

#Initialize Q matrix
Q=np.ones((Rows,Columns))

#Q-learning variables
alpha=0.5
gamma=0.5
epsilon=0.15
goalCounter=0
Contador=0


#function to choose the Action
def ChooseAction (Columns,Q,state):

    if np.random.uniform() < epsilon:
        rand_action=np.random.permutation(Columns)
        action=rand_action[1] #current action
        F=Actions[action]
        max_index=1
    # if not select max action in Qtable (act greedy)
    else:
        QMax=max(Q[state]) 
        max_indices=np.where(Q[state]==QMax)[0] # Identify all indexes where Q equals max
        n_hits=len(max_indices) # Number of hits
        max_index=int(max_indices[random.randint(0, n_hits-1)]) # If many hits, choose randomly
        F=Actions[max_index]

    return F, max_index

#function to apply the dynamic model
def ActionToState(yinit, ts, pivot_x, pivot_y=0.0, is_acceleration=False, l=1.0, g=9.8, d=0.0, h=1e-4, **kwargs):
    """Returns the timeseries of a simulated non inertial pendulum

    Parameters:
    yinit: initial conditions (th, w)
    ts: integration times
    l: the pendulum's length
    g: the local acceleration of gravity
    d: the damping constant
    pivot_x: the horizontal position of the pivot
    pivot_y: the vertical position of the pivot
    is_acceleration: set to True to input pivot accelerations instead of positions
    h: numerical step for computing numerical derivatives
    **kwargs: odeint keyword arguments

    Returns:
    sol: the simulation's timeseries sol[:, 0] = ths, sol[:, 1] = ws
    """

    ## Set the problem
    f = lambda state, t : dpendulum(state, t, pivot_x, pivot_y, is_acceleration, l, g, d, h)

    ## Solve it
    sol = odeint(f, yinit, ts, **kwargs)

    return sol, pivot_x
    

#BEGINNING of the q-learning algorithm
for episode in range(1,200000):
    # initial state
    #angle=0.20
    #omega=0.50
    #state=55*200 + 150 
    state=11150
    yinit = (0.20, 0.50)

    #Q-learning algorithm
    print("episode",episode) #check
  
    for i in range(1,300):

        ## Choose sometimes the Force randomly
        F,max_index = ChooseAction(Columns, Q, state)

        pivot_x = pivot_x + F

        #update the dynamic model
        sol[i] = ActionToState (yinit, 0.1, pivot_x, pivot_y=0.0, is_acceleration=False, l=1.0, g=9.8, d=0.0)#, h=1e-4, **kwargs)
                     
        #do the loop and calculate the reward
        rounded_angle=round(sol[i,0],2)  #round the angle, two decimals
        rounded_omega=round(sol[i,1],2)  #round the omega, two decimals  

        #calculate which is my new state
        index_1=np.where(ANGLES==rounded_angle)
        index_2=np.where(ANGULAR_VEL==rounded_omega)
        index_1=int(index_1[0])
        index_2=int(index_2[0])

        state=n_speeds*index_1 + index_2  #new state in Q matrix
        QMax=max(Q[state])  #selects the highest value of the row
          

        #REWARD
        A1=math.exp(-abs(rounded_angle-Final_angle)/(0.1*n_pos))
        A2=math.exp(-abs(rounded_omega-Final_omega)/(0.1*14))
        Reward=A1*A2*1000000  #takes into account pos and vel

        #Q VALUE update
        Q[state,max_index]=Q[state,max_index] + alpha*(Reward + gamma*(QMax - Q[state,max_index]))  #update Q value
                       

        #checking
        if (rounded_angle<=Final_angle+0.05 and rounded_angle>=Final_angle-0.05):
            print("entra")
            goalCounter=goalCounter+1
            if (rounded_omega<=Final_omega+0.02 and rounded_omega==Final_omega-0.02):
                Contador=Contador +1  #counter of successful hits
                    
                #saving of successful data
                
                state=11150 #reinitialize
                break

            else:
                break
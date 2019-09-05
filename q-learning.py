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
ANGLES= ANGLES.round(decimals=2)

#SPEEDS are discretized -1 rad/s, -0.99, -0.98 ... 0.98, 0,99 1 rad/s.
n_speeds=201
ANGULAR_VEL=np.linspace(-1,1,n_speeds)
ANGULAR_VEL= ANGULAR_VEL.round(decimals=2)

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
sol=np.zeros((1000,2))


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


#BEGINNING of the q-learning algorithm
for episode in range(1,200000):
    # initial state
    #angle=0.20
    #omega=0.50
    #state=56*201 + 151 
    state_=11407 #this state represents the initial state and angle
    yinit = (0.20, 0.50) #theta, omega
    pivot_x=10 #initial pivot_x position

    #Q-learning algorithm
    print("episode",episode) #check
  
    for i in range(1,300):

        ## Choose sometimes the Force randomly
        F,max_index = ChooseAction(Columns, Q, state_)

        pivot_x = pivot_x + F
        t=np.linspace(0+0.1*i,0.1+0.1*i,2)
        #update the dynamic model
        sol = pendulum (yinit, t, pivot_x, pivot_y=0.0, is_acceleration=False, l=1.0, g=9.8, d=0.0)#, h=1e-4, **kwargs)
                     
        #do the loop and calculate the reward
        rounded_angle=round(sol[1,0],2)  #round the angle, two decimals
        rounded_omega=round(sol[1,1],2)  #round the omega, two decimals  

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

        #calculate the new (angle,omega) conditions
        new_angle=state // n_speeds
        new_omega=state % n_speeds
        yinit=(ANGLES[new_angle],ANGULAR_VEL[new_omega])
                     

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

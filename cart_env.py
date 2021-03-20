from scipy.integrate import solve_ivp
from numpy import sin, cos
import numpy as np
import polytope as pt



def cart_ode(t, x, v, w):
    return [v * cos(x[2]), v * sin(x[2]), w]


class Cart(object):
    def __init__(self, x=0, y=0, width=0.1, lf=0.04, lr=0.16, theta=0):
        self.x = x
        self.y = y
        self.theta = theta
        self.lf = lf
        self.lr = lr
        self.width = width

    '''
    step(self,v,w,time) updates the state of the cart

    '''

    def step(self, v, w, time):
        y0 = [self.x, self.y, self.theta]
        sol = solve_ivp(cart_ode, [0, time],
                        y0,
                        method='RK45',
                        args=(v, w),
                        rtol=1e-8)
        self.x = sol.y[0, :][-1]
        self.y = sol.y[1, :][-1]
        self.theta = sol.y[2, :][-1]

    '''
    corners_position(self) computes the 4 corners of the cart
    corners_postion returns a 2d-array "corners"
    corners[0] represent the x-coordinates of the corners
    corners[1] represent the y-coordinates of the corners
    '''

    def corners_position(self):
        corner1 = np.array([self.lf, self.width / 2])
        corner2 = np.array([self.lf, -self.width / 2])
        corner3 = np.array([-self.lr, -self.width / 2])
        corner4 = np.array([-self.lr, self.width / 2])

        corners = np.vstack([corner1, corner2, corner3, corner4])
        corners = corners.T

        # translation and rotation
        translation = np.array([[self.x], [self.y]])
        rotation_matrix = np.array([[cos(self.theta), -sin(self.theta)],
                                    [sin(self.theta),
                                     cos(self.theta)]])
        corners = rotation_matrix @ corners + translation

        return corners


class CartEnv(object):
    def __init__(self, step_time=0.01):
        self.cart = Cart()
        self.obstacles = list()  #list of polytopes
        self.cart_poly = pt.qhull(self.cart.corners_position().T)
        self.goal = pt.qhull(np.array([[1.3, 1.3], [1.3, 2], [2, 1.3], [2,2]]))
        self.frame = pt.qhull(np.array([[-0.5, -0.5], [2, -0.5], [-0.5, 2], [2, 2]]))
        self.step_time = step_time

        print('init successful')

    # the following functions update the data of this class

    def update_cart_polytope(self):
        self.cart_poly = pt.qhull(self.cart.corners_position().T)

    def set_goal(self, corners):
        self.goal = pt.qhull(np.asarray(corners))

    def set_frame(self, corners):
        self.frame = pt.qhull(np.asarray(corners))

    def add_obstacle(self, corners):  #2d array, or 2d list
        p1 = pt.qhull(np.asarray(corners))
        self.obstacles.append(p1)

    # the following check functions check the polytope conditions

    def check_goal(self):
        A = self.goal.A
        b = self.goal.b
        for point in pt.extreme(self.cart_poly):
            #if there is at least one extreme of the cart is out of the goal area, return false.
            if (not (np.all(A @ point - b <= 0))):
                return False
        return True

    def check_frame(self):
        A = self.frame.A
        b = self.frame.b
        for point in pt.extreme(self.cart_poly):
            #if there is at least one extreme of the cart is out of the frame area, return false.
            if (not (np.all(A @ point - b <= 0))):
                return False
        return True

    def check_crash(self):  # Ture: cart hasn't crash into the obstacles.
        for o in self.obstacles:

            #check whether the corners of the cart are in the obstacle
            A = o.A
            b = o.b
            for point in pt.extreme(self.cart_poly):
                if (np.all(A @ point - b <= 0)):
                    return False

            #check corners of the obstacle is in the car
            A = self.cart_poly.A
            b = self.cart_poly.b
            for point in pt.extreme(o):
                if (np.all(A @ point - b <= 0)):
                    return False
        return True

    # step(self, action) accept an action and update the state of the cart  
    # after updating the state, it check the obstacles, goal, frame
    # it return the next state, reward, whether this epoch is done, the information of terminal
    # action 0, move forward
    # action 1, turn left
    # action 2, turn right
    def step(self, action):
        if (action==0):
            v=0.5
            w=0
        
        if (action==1):
            v=0
            w=0.5

        if (action==2):
            v=0
            w=-0.5

        self.cart.step(v, w, self.step_time)
        next_state = [self.cart.x, self.cart.y, self.cart.theta]
        self.update_cart_polytope()
        done = False
        info = []
        # reward = 0 
        reward= -0.01

        if (self.check_crash() == False):
            done = True
            info.append('crashed into obstacle')
            reward = -50

        if (self.check_frame() == False):
            done = True
            info.append('out of frame')
            reward = -50

        if (self.check_goal() == True):
            done = True
            info.append('reach the goal')
            reward = 1000


        return next_state, reward, done, info

    def reset(self):
        self.cart.x = 0
        self.cart.y = 0
        self.cart.theta = 0
        return [0,0,0]

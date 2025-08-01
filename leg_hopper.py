import pickle
import scipy.integrate
from sympy.physics.mechanics import *
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import matplotlib.animation as animation
from enum import Enum
from math import floor


class LegHopperModel:
    class Mode(Enum):
        Air = 1
        Constrained = 2

    def __init__(self, gravity: float, side_mass: float, 
               middle_mass: float, horizontal_bar_len: float):
        self.gravity = gravity
        self.side_mass = side_mass
        self.middle_mass = middle_mass
        self.horizontal_bar_len = horizontal_bar_len
        self.state = np.array([0,0,0.6,np.pi/2,0,0,0,0,0,0,0,0])
        self.mode = LegHopperModel.Mode.Constrained
        self.time = 0.0

    def generate_equations(self):
        g, m, m2, t, l= sp.symbols('g,m,m2,t,l')
        x,y,q1,q2,theta = dynamicsymbols('x y q1 q2 theta')
        dx,dy,dq1,dq2,dtheta = sp.symbols('dx dy dq1 dq2 dtheta')
        ddx,ddy,ddq1,ddq2,ddtheta = sp.symbols('ddx ddy ddq1 ddq2 ddtheta')
        lam1, lam2 = sp.symbols('lam1, lam2')

        vars = sp.Matrix([x,y,q1,q2,theta])
        dvars = sp.Matrix([dx,dy,dq1,dq2,dtheta])

        p1 = sp.Matrix([x-sp.cos(theta)*l, y-sp.sin(theta)*l])
        p1d = sp.diff(p1, t)
        p2 = sp.Matrix([x+sp.cos(theta)*l, y+sp.sin(theta)*l])
        p2d = sp.diff(p2, t)
        p3 = sp.Matrix([x-sp.cos(q2+theta)*q1*0.5, y-sp.sin(q2+theta)*q1*0.5])
        p3d = sp.diff(p3, t)

        # point of feet contact
        pc = sp.Matrix([x-sp.cos(q2+theta)*q1, y-sp.sin(q2+theta)*q1])
        self.pc_func = sp.lambdify([x,y,q1,q2,theta], pc, 'numpy')
        pcd = sp.diff(pc, t)

        # Kinetic and potential energy
        K = 0.5 * m * (p1d.transpose() * p1d + p2d.transpose() * p2d) +\
            0.5 * m2 * p3d.transpose() * p3d
        K = K[0]
        V = g * m * (p1[1] + p2[1]) + g * m2 * p3[1]
        # constraints equations fc=0
        fc1 = pc[0]  
        fc2 = pc[1] 
        # Jacobian
        J = pc.jacobian([x,y,q1,q2,theta])
        # Lagrangian
        L = sp.simplify(K-V)

        # replace vars
        replace_dic = {g: 9, m: 0.3, m2: 0.2, l: 0.2}
        L = L.subs(replace_dic)
        fc1 = fc1.subs(replace_dic)
        fc2 = fc2.subs(replace_dic)
        J = J.subs(replace_dic)

        # Lagrnage-Euler equations
        eqs = []
        for qi in [x,y,q1,q2,theta]:
            eq = sp.diff(L, qi) - sp.diff(sp.diff(L, sp.diff(qi, t)), t) \
                + sp.diff(fc1, qi) * lam1 + sp.diff(fc2, qi) * lam2
            eqs.append(eq)
        eqs.append(sp.diff(fc1,t,2))
        eqs.append(sp.diff(fc2,t,2))

        sub_dic = {
            sp.diff(x, t, 2): ddx,
            sp.diff(y, t, 2): ddy,
            sp.diff(q1, t, 2): ddq1,
            sp.diff(q2, t, 2): ddq2,
            sp.diff(theta, t, 2): ddtheta,
            sp.diff(x, t): dx,
            sp.diff(y, t): dy,
            sp.diff(q1, t): dq1,
            sp.diff(q2, t): dq2,
            sp.diff(theta, t): dtheta}
        eqs = [eq.subs(sub_dic) for eq in eqs]
        pcd = pcd.subs(sub_dic)
        self.pcd_func = sp.lambdify([x,y,q1,q2,theta,dx,dy,dq1,dq2,dtheta], pcd, 'numpy')

        # Equations of motion without any constraints 
        eqs_nc = [eq.subs({lam1: 0, lam2: 0}) for eq in eqs]
        eqs_nc = eqs_nc[:-2]

        # ---------------- Non-constrained model (no contact) ----------------
        # convert differential equation into lineary set of equation
        sol = sp.linear_eq_to_matrix(eqs_nc, [ddx, ddy, ddq1,  ddq2,  ddtheta])
        # transform from 2nd order ot 1st order eqs
        q_size = dvars.shape[0] # num of variables (dof)
        MM_nc_sym = sp.Matrix([[sp.eye(q_size),sp.zeros(q_size, sol[0].shape[1])],
                            [sp.zeros(sol[0].shape[1], q_size), sol[0]]])
        F_nc_sym = sp.Matrix([[dvars],[sol[1]]])
        self.MM_nc_func = sp.lambdify([x,y,q1,q2,theta,dx,dy,dq1,dq2,dtheta], MM_nc_sym, 'numpy')
        self.F_nc_func = sp.lambdify([x,y,q1,q2,theta,dx,dy,dq1,dq2,dtheta], F_nc_sym, 'numpy')

        # ---------------- Constrained model (feet touching ground) ----------------
        # convert differential equation into lineary set of equation
        sol = sp.linear_eq_to_matrix(eqs, [ddx, ddy, ddq1,  ddq2,  ddtheta, lam1, lam2])
        # transform from 2nd order ot 1st order eqs
        q_size = dvars.shape[0] # num of variables (dof)
        MM_sym = sp.Matrix([[sp.eye(q_size),sp.zeros(q_size, sol[0].shape[1])],
                            [sp.zeros(sol[0].shape[1], q_size), sol[0]]])
        F_sym = sp.Matrix([[dvars],[sol[1]]])
        self.MM_func = sp.lambdify([x,y,q1,q2,theta,dx,dy,dq1,dq2,dtheta], MM_sym, 'numpy')
        self.F_func = sp.lambdify([x,y,q1,q2,theta,dx,dy,dq1,dq2,dtheta], F_sym, 'numpy')

        # velocity stopper structures
        MMat = sol[0][:q_size, :q_size]
        self.MMat_func = sp.lambdify(vars, MMat, 'numpy')
        self.JMat_func = sp.lambdify(vars, J, 'numpy')

    def reset(self, state: np.ndarray):
        self.set_state(state)
        self.time = 0.0

    def set_state(self, state: np.ndarray):
        array_len = state.shape[0]
        if array_len < 10 or array_len > 12:
            raise ValueError("Wrong number of state variables. Expected from 10 to 12.")
        self.state[:array_len] = state

    def get_feet_position(self):
        y = self.state
        return self.pc_func(y[0],y[1],y[2],y[3],y[4])
    
    def get_feet_velocity(self):
        y = self.state
        return self.pcd_func(y[0],y[1],y[2],y[3],y[4],y[5],y[6],y[7],y[8],y[9])

    def advance(self, dt: np.float64, control: np.ndarray):
        y = self.state
        self.time += dt
        if self.mode == LegHopperModel.Mode.Air:
            MM = self.MM_nc_func(y[0],y[1],y[2],y[3],y[4],y[5],y[6],y[7],y[8],y[9])
            FF = self.F_nc_func(y[0],y[1],y[2],y[3],y[4],y[5],y[6],y[7],y[8],y[9])
            FF[7:9] += control.reshape((2,1))
            result = np.linalg.solve(MM,FF)
            self.state[:10] = y[:10] + np.squeeze(result[:10]) * dt
            self.state[-3:-1] = np.array([0,0])
            pc = self.get_feet_position()
            pcd = self.get_feet_velocity()
            if pc[1] <= -0.6 and pcd[1] <= 0:
                #self.handle_contact_impact()
                self.mode = LegHopperModel.Mode.Constrained
                
        elif self.mode == LegHopperModel.Mode.Constrained:
            self.handle_contact_impact()
            MM = self.MM_func(y[0],y[1],y[2],y[3],y[4],y[5],y[6],y[7],y[8],y[9])
            FF = self.F_func(y[0],y[1],y[2],y[3],y[4],y[5],y[6],y[7],y[8],y[9])
            # Torque for nth variable need to be offset by velocity variables (5+n)
            FF[7:9] += control.reshape((2,1))
            #FF[7] -= 40
            result = np.linalg.solve(MM,FF)
            self.state[:10] = y[:10] + np.squeeze(result[:10]) * dt
            self.state[-3:-1] = np.squeeze(result[10:12])
            pc = self.get_feet_position()
            if result[-1] < -0.0 or pc[1] > -0.6:
                self.mode = LegHopperModel.Mode.Air
                
    def handle_contact_impact(self):
        y = self.state
        Ms = self.MMat_func(y[0],y[1],y[2],y[3],y[4])
        Js = self.JMat_func(y[0],y[1],y[2],y[3],y[4])
        q_old = y[5 : 10]
        impactSigma = np.matmul(np.matmul(Js, np.linalg.inv(Ms)), np.transpose(Js))
        impactSigma = np.matmul(np.matmul(-np.linalg.inv(impactSigma), Js), q_old)
        q_new = q_old + np.matmul(np.matmul(np.linalg.inv(Ms), np.transpose(Js)), impactSigma)
        self.state[5 : 10] = q_new

if __name__ == "__main__":
    robot = LegHopperModel(gravity=9, side_mass=0.3, 
                        middle_mass=0.2, horizontal_bar_len=0.2)
    robot.generate_equations()
    robot.set_state(np.array([0,0.1,0.6,np.pi/2,0,0,0,0,0,0,0,0]))

    sim_time = 1.3
    dt = 0.001
    n_samples = floor(sim_time/dt)

    t = 0
    sola = np.zeros(shape=(n_samples, 12), dtype='float64')
    sola_t = np.zeros(n_samples)

    for k in range(1,n_samples):
        print(t)
        t = t + dt
        sola_t[k] = t

        state = robot.state

        q1_target = 1 if k < 200 else 0.6
        Fq1 = (state[2] - q1_target) * 80 + 0.6 * state[7]
        Fq2 = (state[3] - np.pi/2 + 0.1) * 80 + 0.0 * state[8]
        #Fq2 = 0
        robot.advance(dt, np.array([Fq1,Fq2]))

        print(robot.state)
        sola[k,:] = robot.state

        


    # Quick plotting data
    plt.plot(sola_t, sola[:,0])
    plt.plot(sola_t, sola[:,1])
    plt.show()


    # Animation/Visualization
    fig, ax = plt.subplots()
    line, = ax.plot([], [], 'o-', lw=2)
    line2, = ax.plot([], [], 'o-', lw=2)
    ax.set_xlim(left=-0.7,right=0.7)
    ax.set_ylim(top=1.5,bottom=-1)
    ax.set_aspect('equal')

    lp = 0.2

    def animate(i):
        x0 = sola[i,0]
        y0 = sola[i,1]
        q1 = sola[i,2]
        q2 = sola[i,3]
        theta = sola[i,4]
        
        x1 = x0-np.cos(theta)*lp
        y1 = y0-np.sin(theta)*lp
        x2 = x0+np.cos(theta)*lp
        y2 = y0+np.sin(theta)*lp

        x3 = x0 - np.cos(q2+theta)*q1
        y3 = y0 - np.sin(q2+theta)*q1

        line.set_data(np.array([[x1,x2],[y1,y2]]))
        line2.set_data(np.array([[x0, x3],[y0, y3]]))
        return (line,line2)

    ani = animation.FuncAnimation(
        fig, animate, len(sola[:,0]), interval=10, blit=False)
    plt.show()
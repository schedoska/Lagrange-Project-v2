import pickle
import scipy.integrate
from sympy.physics.mechanics import *
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import matplotlib.animation as animation


sp.init_printing()

# Defining symbols
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


K = 0.5 * m * (p1d.transpose() * p1d + p2d.transpose() * p2d) +\
    0.5 * m2 * p3d.transpose() * p3d
K = K[0]
V = g * m * (p1[1] + p2[1]) + g * m2 * p3[1]

# constraints equations fc=0
fc1 = pc[0]  
fc2 = pc[1] 
# Jacobian
J = p1.jacobian([x,y,q1,q2,theta])

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

eqs = [eq.subs({
    sp.diff(x, t, 2): ddx,
    sp.diff(y, t, 2): ddy,
    sp.diff(q1, t, 2): ddq1,
    sp.diff(q2, t, 2): ddq2,
    sp.diff(theta, t, 2): ddtheta,
    sp.diff(x, t): dx,
    sp.diff(y, t): dy,
    sp.diff(q1, t): dq1,
    sp.diff(q2, t): dq2,
    sp.diff(theta, t): dtheta}) for eq in eqs]

# convert differential equation into lineary set of equation
sol = sp.linear_eq_to_matrix(eqs, [ddx, ddy, ddq1,  ddq2,  ddtheta, lam1, lam2])
# transform from 2nd order ot 1st order eqs
q_size = dvars.shape[0] # num of variables (dof)
MM_sym = sp.Matrix([[sp.eye(q_size),sp.zeros(q_size, sol[0].shape[1])],
                    [sp.zeros(sol[0].shape[1], q_size), sol[0]]])
F_sym = sp.Matrix([[dvars],[sol[1]]])
MM_func = sp.lambdify([x,y,q1,q2,theta,dx,dy,dq1,dq2,dtheta], MM_sym, 'numpy')
F_func = sp.lambdify([x,y,q1,q2,theta,dx,dy,dq1,dq2,dtheta], F_sym, 'numpy')

# velocity stopper structures
MMat = sol[0][:q_size, :q_size]
MMat_func = sp.lambdify(vars, MMat, 'numpy')
JMat_func = sp.lambdify(vars, J, 'numpy')


# Temporal Euler solver
y = np.array([0,0,0.6,np.pi/2,0,0,0,0,0,0,0,0],'float64')
sola = np.zeros(shape=(1001,12), dtype='float64')
for t in range(1,400):
    print(t)
    MM = MM_func(y[0],y[1],y[2],y[3],y[4],y[5],y[6],y[7],y[8],y[9])
    FF = F_func(y[0],y[1],y[2],y[3],y[4],y[5],y[6],y[7],y[8],y[9])
    # Torque for nth variable need to be offset by velocity variables (5+n)
    FF[7] += (y[1]) * 50
    result = np.linalg.solve(MM,FF)
    print(np.squeeze(result))
    y = y + np.squeeze(result) * np.float64(0.005) 
    sola[t,:] = y



# Animation/Visualization
fig, ax = plt.subplots()
line, = ax.plot([], [], 'o-', lw=2)
line2, = ax.plot([], [], 'o-', lw=2)
ax.set_xlim(left=-0.7,right=0.7)
ax.set_ylim(top=.5,bottom=-1.5)
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
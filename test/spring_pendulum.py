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
lam = dynamicsymbols('lam')

vars = sp.Matrix([x,y,q1,q2,theta])
dvars = sp.diff(vars,t)

p1 = sp.Matrix([x-sp.cos(theta)*l, y-sp.sin(theta)*l])
p1d = sp.diff(p1, t)
p2 = sp.Matrix([x+sp.cos(theta)*l, y+sp.sin(theta)*l])
p2d = sp.diff(p2, t)
p3 = sp.Matrix([x-sp.cos(q2+theta)*q1*0.5, y-sp.sin(q2+theta)*q1*0.5])
p3d = sp.diff(p3, t)
# poinet of feet contact
pc = sp.Matrix([x-sp.cos(q2+theta)*q1, y-sp.sin(q2+theta)*q1])
pcd = sp.diff(pc, t)
pcdd = sp.diff(pc, t, 2)


K = 0.5 * m * (p1d.transpose() * p1d + p2d.transpose() * p2d) +\
    0.5 * m2 * p3d.transpose() * p3d
K = K[0]
V = g * m * (p1[1] + p2[1]) + g * m2 * p3[1]

# constraints equations fc=0
fc1 = pcdd[0]  
fc2 = pcdd[1] 






# Lagrangian
L = sp.simplify(K-V)

# replace vars
replace_dic = {g: 9, m: 0.3, m2: 0.2, l: 0.2}
L = L.subs(replace_dic)
fc1 = fc1.subs(replace_dic)
fc2 = fc2.subs(replace_dic)


args_list = vars.row_insert(vars.shape[0], dvars)
#args_list = args_list.row_insert(args_list.shape[0], sp.Matrix([xc,yc]))
print(args_list)

lag = LagrangesMethod(L, [x,y,q1,q2,theta], nonhol_coneqs=[fc1,fc2])
eoms = lag.form_lagranges_equations()
MM = lag.mass_matrix_full
F = lag.forcing_full
MM_func = sp.lambdify(args_list, MM, 'numpy')
F_func = sp.lambdify(args_list, sp.simplify(F), 'numpy')



def system_model(t, q):
    print(t)
    MM = MM_func(q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9])
    FF = F_func(q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9])
    FF[2] -= 0
    #result = np.linalg.solve(MM,FF)
    m=np.linalg.lstsq(MM, FF)
    #print(result)
    #print(np.squeeze(m[0]))
    #quit()
    #print(result)
    print(np.squeeze(m[0]))
    return np.squeeze(m[0])

q0 = [0,0,0.6,np.pi/2,0.0,0,0,0,0,0,0.0,1]
sola = solve_ivp(system_model, [0, 0.3], y0=q0, method="RK45", max_step=0.001, atol = 1, rtol = 1)



plt.plot(sola.t, sola.y[0,:])
plt.plot(sola.t, sola.y[2,:])
plt.show()



theta_arr = np.linspace(-0.2,0.2,100)
q2_arr = np.linspace(np.pi/2,2,100)
#q2_arr = np.ones(1000) * np.pi/2

fig, ax = plt.subplots()
line, = ax.plot([], [], 'o-', lw=2)
line2, = ax.plot([], [], 'o-', lw=2)
ax.set_xlim(left=-0.7,right=0.7)
ax.set_ylim(top=.5,bottom=-1.5)
ax.set_aspect('equal')

x0=y0=0
lp = 0.2

def animate(i):
    x0 = sola.y[0,i]
    y0 = sola.y[1,i]
    #theta = theta_arr[i]
    # q2 = q2_arr[i]
    theta = sola.y[4,i]
    q2 = sola.y[3,i]
    q1 = sola.y[2,i]

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
    fig, animate, len(sola.y[0,:]), interval=10, blit=False)
plt.show()



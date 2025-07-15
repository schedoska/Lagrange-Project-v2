import scipy.integrate
from sympy.physics.mechanics import *
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import matplotlib.animation as animation


sp.init_printing()

# Defining symbols
l1, l2, g, m, t = sp.symbols('l1,l2,g,m,t')
x,y,q1,q2 = dynamicsymbols('x y q1 q2')
lam = dynamicsymbols('lam')

vars = sp.Matrix([x,y,q1,q2])
dvars = sp.diff(vars,t)
ddvars = sp.diff(vars,t,t)

p1 = sp.Matrix([-sp.cos(q1) * l1 + x, -sp.sin(q1) * l1 + y])
p2 = sp.Matrix([sp.cos(q1) * l1 + x, sp.sin(q1) * l1 + y])
p3 = sp.Matrix([x - l2 * 0.5 * sp.sin(sp.pi/2-q2-q1),
                y - l2 * 0.5 * sp.cos(sp.pi/2-q2-q1)])
p1d = sp.diff(p1, t)
p2d = sp.diff(p2, t)
p3d = sp.diff(p3, t)

K = 0.5 * m * (p1d.transpose()*p1d + p2d.transpose()*p2d + 0.1*p3d.transpose()*p3d)
K = K[0]
V = m * g * (p1[1] + p2[1] + 0.1*p3[1])

# constraints
fc = x**2 + y**2 - l2**2

# Lagrangian
L = sp.simplify(K-V)
L = L.subs({g: 9, m: 1, l1: 0.2, l2:0.6})

eqs = []
for qi in [x,y,q1,q2]:
    eq = sp.diff(L, qi) - sp.diff(sp.diff(L, sp.diff(qi, t)), t) + sp.diff(fc, qi) * lam
    eqs.append(eq)
eqs.append(sp.diff(fc,t,t))


to_solve = ddvars
to_solve = to_solve.row_insert(to_solve.shape[0], sp.Matrix([lam]))
sol = sp.solve(eqs, to_solve, simplify=False)


f_list = []
args_mat = vars.row_insert(vars.shape[0], dvars)
for k, v in sol.items():
    f_list.append(sp.lambdify(args_mat, sol[k], 'numpy'))

print(f_list)
print(args_mat)

def system_model(t, q):
    return [q[4], 
            q[5],
            q[6],
            q[7],
            f_list[0](q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7]),
            f_list[1](q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7]),
            f_list[2](q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7]),
            f_list[3](q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7])]


sola = solve_ivp(system_model, [0, 100], y0=[0,0.6,0,sp.pi/2,0,0,0.3,0], method='DOP853',
                 t_eval=np.arange(0,100,0.01))






fig, ax = plt.subplots()
line, = ax.plot([], [], 'o-', lw=2)
line2, = ax.plot([], [], 'o-', lw=2)
ax.set_xlim(left=-0.7,right=0.7)
ax.set_ylim(top=1.5,bottom=-.5)
ax.set_aspect('equal')

l1p = 0.2
l2p = 0.6
xp = yp = 0


def animate(i):
    q1 = sola.y[2,i]
    q2 = sola.y[3,i]
    x1 = -np.cos(q1) * l1p + sola.y[0,i]
    x2 = np.cos(q1) * l1p + sola.y[0,i]
    y1 = -np.sin(q1) * l1p + sola.y[1,i]
    y2 = np.sin(q1) * l1p + sola.y[1,i]

    line.set_data(np.array([[x1, x2],[y1, y2]]))
    line2.set_data(np.array([[sola.y[0,i], 0],[sola.y[1,i], 0]]))

    return (line, line2)

ani = animation.FuncAnimation(
    fig, animate, len(sola.y[0,:]), interval=60, blit=False)
plt.show()
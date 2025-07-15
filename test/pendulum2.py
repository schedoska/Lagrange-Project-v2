import scipy.integrate
from sympy.physics.mechanics import *
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import matplotlib.animation as animation


sp.init_printing()

# Defining symbols
l, g, m, t = sp.symbols('R,g,m,t')
x, y, q = dynamicsymbols('x y q')

dx, dy, dq = sp.symbols('dx dy dq')
ddx, ddy, ddq = sp.symbols('ddx ddy ddq')
lam1, lam2 = sp.symbols('lam1, lam2')

dvars = sp.Matrix([dx, dy, dq])

p1 = sp.Matrix([x - sp.cos(q) * l, y + sp.sin(q) * l])
p1d = sp.diff(p1, t)
p2 = sp.Matrix([x + sp.cos(q) * l, y - sp.sin(q) * l])
p2d = sp.diff(p2, t)

K = 0.5 * m * (p1d.transpose() * p1d + p2d.transpose() * p2d)
K = K[0]
V = g * m * (p1[1] + p2[1])

fc1 = p1[0]
fc2 = p1[1]
# Jacobian
J = p1.jacobian([x,y,q])

# Lagrangian
L = sp.simplify(K-V)

# replace vars
replace_dic = {g: 9, m: 0.3, l: 0.5}
L = L.subs(replace_dic)
fc1 = fc1.subs(replace_dic)
fc2 = fc2.subs(replace_dic)
J = J.subs(replace_dic)


eqs = []
for qi in [x,y,q]:
    eq = sp.diff(L, qi) - sp.diff(sp.diff(L, sp.diff(qi, t)), t) \
        + sp.diff(fc1, qi) * lam1 + sp.diff(fc2, qi) * lam2
    eqs.append(sp.simplify(eq))
eqs.append(sp.simplify(sp.diff(fc1,t,2)))
eqs.append(sp.simplify(sp.diff(fc2,t,2)))
#eqs.append(fc2)
#eqs.append(fc1)

eqs = [eq.subs({
    sp.diff(x, t, 2): ddx,
    sp.diff(y, t, 2): ddy,
    sp.diff(q, t, 2): ddq,
    sp.diff(x, t): dx,
    sp.diff(y, t): dy,
    sp.diff(q, t): dq,}) for eq in eqs]



sol = sp.linear_eq_to_matrix(eqs, [ddx, ddy, ddq, lam1, lam2])
# transform from 2nd order ot 1st order eqs
q_size = 3 # num of changabkle variables
MM_sym = sp.Matrix([[sp.eye(q_size),sp.zeros(q_size, sol[0].shape[1])],
                    [sp.zeros(sol[0].shape[1], q_size), sol[0]]])
F_sym = sp.Matrix([[dvars],[sol[1]]])

MM_func = sp.lambdify([x,y,q,dx,dy,dq], MM_sym, 'numpy')
F_func = sp.lambdify([x,y,q,dx,dy,dq], F_sym, 'numpy')

# velocity stopper
MMat = sol[0][:3, :3]
MMat_func = sp.lambdify([x,y,q], MMat, 'numpy')
JMat_func = sp.lambdify([x,y,q], J, 'numpy')



def system_model(t, q):
    print(t)

    if t > 0.5 and system_model.flag == False:
        Ms = MMat_func(q[0],q[1],q[2])
        Js = JMat_func(q[0],q[1],q[2])
        q_old = q[3:6]
        impactSigma = np.matmul(np.matmul(Js, np.linalg.inv(Ms)), np.transpose(Js))
        impactSigma = np.matmul(np.matmul(-np.linalg.inv(impactSigma), Js), q_old)
        q_new = q_old + np.matmul(np.matmul(np.linalg.inv(Ms), np.transpose(Js)), impactSigma)
        q[3:6] = q_new

    MM = MM_func(q[0],q[1],q[2],q[3],q[4],q[5])
    FF = F_func(q[0],q[1],q[2],q[3],q[4],q[5])
    result = np.linalg.solve(MM,FF)
    return np.squeeze(result)
system_model.flag = False

q0 = [0,0,0,-3,0,0,0,0]
#sola = solve_ivp(system_model, [0, 1], y0=q0, method='Radau', max_step=0.001)


y = np.array([0,0,0,0,0.3,0,0,0],'float64')
sola = np.zeros(shape=(1001,6), dtype='float64')
for t in range(1,1000):
    print(t)
    if t > 500 and system_model.flag == False:
        Ms = MMat_func(y[0],y[1],y[2])
        Js = JMat_func(y[0],y[1],y[2])
        q_old = y[3:6]
        impactSigma = np.matmul(np.matmul(Js, np.linalg.inv(Ms)), np.transpose(Js))
        impactSigma = np.matmul(np.matmul(-np.linalg.inv(impactSigma), Js), q_old)
        q_new = q_old + np.matmul(np.matmul(np.linalg.inv(Ms), np.transpose(Js)), impactSigma)
        y[3:6] = q_new
    MM = MM_func(y[0],y[1],y[2],y[3],y[4],y[5])
    FF = F_func(y[0],y[1],y[2],y[3],y[4],y[5])
    result = np.linalg.lstsq(MM,FF)[0]
    print(y)
    print(np.squeeze(result))
    y = y + np.squeeze(result) * np.float64(0.005) 
    print(y)
    sola[t,:] = y[:6]






q_arr = np.linspace(0,0.5,100)

fig, ax = plt.subplots()
line, = ax.plot([], [], 'o-', lw=2)
ax.set_xlim(left=-1.2,right=1.2)
ax.set_ylim(top=1,bottom=-1)
ax.set_aspect('equal')

l = 0.5

def animate(i):
    qa = sola[i,2]
    xa = sola[i,0]
    ya = sola[i,1]
    p1x = xa - np.cos(qa) * l
    p1y = ya + np.sin(qa) * l
    p2x = xa + np.cos(qa) * l
    p2y = ya - np.sin(qa) * l

    line.set_data(np.array([[p1x, p2x],[p1y, p2y]]))
    return (line)

ani = animation.FuncAnimation(
    fig, animate, len(sola[:,0]), interval=5, blit=False)
plt.show()
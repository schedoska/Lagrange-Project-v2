import scipy.integrate
from sympy.physics.mechanics import *
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import matplotlib.animation as animation


sp.init_printing()

# Defining symbols
R, g, m, t = sp.symbols('R,g,m,t')
x, y, q = dynamicsymbols('x y q')
lam = dynamicsymbols('lam')


vars = sp.Matrix([x,y,q])
dvars = sp.diff(vars,t)
ddvars = sp.diff(vars,t,t)

p1 = sp.Matrix([x + sp.sin(q) * R, y + sp.cos(q) * R])
p1d = sp.diff(p1, t)
p2 = sp.Matrix([x - sp.sin(q) * R, y - sp.cos(q) * R])
p2d = sp.diff(p2, t)

K = 0.5 * m * (p1d.transpose() * p1d + p2d.transpose() * p2d)
K = K[0]
V = g * m * (p1[1] + p2[1])


fc1 = p1d[0]
fc2 = p1d[1]

# Lagrangian
L = sp.simplify(K-V)

# replace vars
replace_dic = {g: 9, m: 0.3, R: 0.4}
L = L.subs(replace_dic)
fc1 = fc1.subs(replace_dic)
fc2 = fc2.subs(replace_dic)


lag = LagrangesMethod(L, [x, y, q], nonhol_coneqs=[fc1,fc2])
eoms = lag.form_lagranges_equations()
solv = lag.rhs()

print(lag.forcing_full)
quit()



func = sp.lambdify([x,y,q,sp.diff(x,t),sp.diff(y,t),sp.diff(q,t)], solv, 'numpy')
print(func(1,1,1,1,1,1))



# model with contraints
eqs = []
for qi in [x,y,q]:
    eq = sp.diff(L, qi) - sp.diff(sp.diff(L, sp.diff(qi, t)), t) + sp.diff(fc1, qi) * lam
    eqs.append(sp.simplify(eq))
eqs.append(sp.simplify(sp.diff(fc1,t)))


to_solve = ddvars
to_solve = to_solve.row_insert(to_solve.shape[0], sp.Matrix([lam]))
sol = sp.solve(eqs, to_solve, simplify=False)



f_list = []
args_mat = vars.row_insert(vars.shape[0], dvars)
for k, v in sol.items():
    f_list.append(sp.lambdify(args_mat, sol[k], 'numpy'))

print(f_list)
print(args_mat)




MM = lag.mass_matrix_full
F = lag.forcing_full
MM_func = sp.lambdify([x,y,q,sp.diff(x,t),sp.diff(y,t),sp.diff(q,t)], MM, 'numpy')
F_func = sp.lambdify([x,y,q,sp.diff(x,t),sp.diff(y,t),sp.diff(q,t)], F, 'numpy')




def system_model(t, q):
    print(t)
    # print(np.squeeze(func(q[0],q[1],q[2],q[3],q[4],q[5])))
    MM = MM_func(q[0],q[1],q[2],q[3],q[4],q[5])
    FF = F_func(q[0],q[1],q[2],q[3],q[4],q[5])
    return np.squeeze(np.linalg.solve(MM,FF))

q0 = [-0.4,0.0,np.pi/2+0.3,0.0001,0.0001,0.001,0.1,0.01]
#sola = solve_ivp(system_model, [0, 2], y0=q0, method="RK45", max_step=0.001, atol = 1, rtol = 1)
#sola = solve_ivp(system_model, [0, 5], y0=q0, method="RK45", max_step=0.004, atol = 1, rtol = 1)
#sola = solve_ivp(system_model, [0, 5], y0=q0, method="Radau", max_step=0.001, atol = 1, rtol = 1)
sola = solve_ivp(system_model, [0, 5], y0=q0, method="DOP853", max_step=0.004)


    



plt.plot(sola.t, sola.y[0,:])
plt.plot(sola.t, sola.y[1,:])
plt.plot(sola.t, sola.y[2,:])
plt.show()



q2_arr = np.linspace(np.pi/2,2,100)

fig, ax = plt.subplots()
line, = ax.plot([], [], 'o-', lw=2)
ax.set_xlim(left=-1.2,right=1.2)
ax.set_ylim(top=1,bottom=-1)
ax.set_aspect('equal')

R0 = 0.4
x0 = 0
y0 = 0

def animate(i):
    print(i) if i % 10 == 0 else 0
    qi = sola.y[2,i]
    x0 = sola.y[0,i]
    y0 = sola.y[1,i]

    x1 = x0 + np.sin(qi) * R0
    y1 = y0 + np.cos(qi) * R0
    x2 = x0 - np.sin(qi) * R0
    y2 = y0 - np.cos(qi) * R0
    line.set_data(np.array([[x1, x2],[y1, y2]]))
    return (line)

ani = animation.FuncAnimation(
    fig, animate, len(sola.y[0,:]), interval=10, blit=False)
plt.show()




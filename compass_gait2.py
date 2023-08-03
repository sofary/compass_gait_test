import numpy as np
from numpy.linalg import inv as inv
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


#Параметры системы
m = 5
M_big = 10
a = 0.5
b = 0.5
l = a + b
bet = m/M_big
mu = a/l
v = b/l 
alpha = 30

g = 9.81
flag = 0
	

def one_leg_phase(t, y): 
    res = [0.1,0,-0.1,0.1]
    res[0] = y[2]
    res[1] = y[3]
    del_y = y[0]-y[1]
    M_matrix = np.array([[bet*mu+1+bet, -bet*v*np.cos(del_y)],[-bet*v*np.cos(del_y), bet*v**2]])
    N_matrix = np.array([[0, -bet*v*np.sin(del_y)*y[1]],[-bet*v*np.sin(del_y)*y[0], 0]])
    G_matrix = np.array([[-(bet*mu+1+bet)*g/l*np.sin(y[0]-alpha)],[bet*(1-mu)*g/l*np.sin(y[1]-alpha)]])
    M_1N = np.dot(inv(M_matrix),N_matrix)
    M_1N_dt = np.dot(M_1N,np.array([[y[2]],[y[3]]]))
    tmp_right_side = -M_1N_dt-G_matrix
    res[2] = float(tmp_right_side[0])
    res[3] = float(tmp_right_side[1])
    if flag > 0:
        fi = y[1]-y[0]
        v_minus = np.array([[-bet*v*mu + (2*bet*v+1)*np.cos(fi), -bet*v*mu],[-bet*v*mu, 0]])
        v_plus = np.array([[bet*v**2 + 1 + bet - bet*mu*np.cos(fi), bet*v**2 - bet*v*np.cos(fi)],[-bet*v*np.cos(fi), bet*v**2]])
        K = np.concatenate([[[0,0],[0,0]],np.dot(inv(v_plus), v_minus)], axis = 1)
        Up_part = np.array([[0,1,0,0],[1,0,0,0]])
        J = np.concatenate([Up_part, K])
        det = np.linalg.det(J)
        for i in range(flag):
           res = [item*(det)**(-1)   for item in res] 
    return res

# def one_leg_phase_(t, y): 
#     res = [0.1,0,-0.1,0.1]
#     res[0] = y[2]
#     res[1] = y[3]
#     del_y = y[0]-y[1]
#     M_matrix = np.array([[bet*mu+1+bet, -bet*v*np.cos(del_y)],[-bet*v*np.cos(del_y), bet*v**2]])
#     N_matrix = np.array([[0, -bet*v*np.sin(del_y)*y[1]],[-bet*v*np.sin(del_y)*y[0], 0]])
#     G_matrix = np.array([[-(bet*mu+1+bet)*g/l*np.sin(y[0]-alpha)],[bet*(1-mu)*g/l*np.sin(y[1]-alpha)]])
#     M_1N = np.dot(inv(M_matrix),N_matrix)
#     M_1N_dt = np.dot(M_1N,np.array([[y[2]],[y[3]]]))
#     tmp_right_side = -M_1N_dt-G_matrix
#     res[2] = float(tmp_right_side[0])
#     res[3] = float(tmp_right_side[1])
#     return res


def hit_detection(t,y):
	return 2*alpha+y[0]+y[1]-4*np.pi

	
#y_0 = [0,0,0,0]
#y_0 = np.array([np.pi/7,np.pi/6,-np.pi/6,np.pi/7])
y_0 = np.array([-23.116814692820412, -23.316814692820415, -0.8, 2.1])
#np.array([0.2 + 2*np.pi + 0.4 - alpha,2*np.pi + 0.4 - alpha, -0.8, 2.1])
t_start = 0
t_end = 4


y_sol_total0 = []
y_sol_total1 = []
y_sol_total2 = []
y_sol_total3 = []
interraotion_t = []
hit_t = []
while True:
    print(0)
    sol = solve_ivp(one_leg_phase, (t_start, t_end), y_0,   events=hit_detection, min_step = 0.01)#, t_eval=np.linspace(t_start, t_end, 100))
    for i in range(len(sol.y[1])-1):
        y_sol_total0.append(sol.y[0][i])
        y_sol_total1.append(sol.y[1][i])
        y_sol_total2.append(sol.y[2][i])
        y_sol_total3.append(sol.y[3][i])
        interraotion_t.append(sol.t[i])

    if len(sol.t_events[0]) < 1:
        break
    #преобразование координат при смене ноги
    print(1)
    flag += 1
    y_0 = sol.y[:, -1]
    y = [0,0,0,0]
    fi = y_0[1]-y_0[0]
    v_minus = np.array([[-bet*v*mu + (2*bet*v+1)*np.cos(fi), -bet*v*mu],[-bet*v*mu, 0]])
    v_plus = np.array([[bet*v**2 + 1 + bet - bet*mu*np.cos(fi), bet*v**2 - bet*v*np.cos(fi)],[-bet*v*np.cos(fi), bet*v**2]])
    K = np.dot(inv(v_plus), v_minus)
    Ky = np.dot(K, np.array([[y_0[2]],[y_0[3]]]))
    y[2] = float(Ky[0])
    y[3] = float(Ky[1])
    y[1] = y_0[0]
    y[0] = y_0[1]

    t_start = sol.t_events[0][-1]
    hit_t.append(t_start)
    sol_ = solve_ivp(one_leg_phase, (t_start, t_end), y,   events=hit_detection, min_step = 0.01)#, t_eval=np.linspace(t_start, t_end, 100))
    #преобразование координат начального условия при смене ноги
    for i in range(len(sol_.y[1])-1):
        y_sol_total0.append(sol_.y[0][i])
        y_sol_total1.append(sol_.y[1][i])
        y_sol_total2.append(sol_.y[2][i])
        y_sol_total3.append(sol_.y[3][i])
        interraotion_t.append(sol_.t[i])
    if len(sol_.t_events[0]) < 1:
        break
    flag += 1
    #преобразование координат начального условия при смене ноги
    y_0 = [0,0,0,0]
    y = sol.y[:, -1]
    fi = y[1]-y[0]
    v_minus = np.array([[-bet*v*mu + (2*bet*v+1)*np.cos(fi), -bet*v*mu],[-bet*v*mu, 0]])
    v_plus = np.array([[bet*v**2 + 1 + bet - bet*mu*np.cos(fi), bet*v**2 - bet*v*np.cos(fi)],[-bet*v*np.cos(fi), bet*v**2]])
    K = inv(np.dot(inv(v_plus), v_minus))
    Ky = np.dot(K,np.array([[y[2]],[y[3]]]))
    y_0[2] = float(Ky[0])
    y_0[3] = float(Ky[1])
    y_0[1] = y[0]
    y_0[0] = y[1]


    #y_0 = sol_.y[:, -1]
    t_start = sol_.t_events[0][-1]
    hit_t.append(t_start)
    print(2)


plt.plot(y_sol_total0)#, y_sol_total2)
plt.ylabel(r'tetta st', fontsize=16)
plt.show()

plt.plot(y_sol_total1)#, y_sol_total3)
plt.ylabel(r'tetta sw', fontsize=16)
plt.show()

plt.plot(y_sol_total0, y_sol_total2)
plt.xlabel(r'tetta st', fontsize=16)
plt.ylabel(r'tetta st dot', fontsize=16)
plt.show()

plt.plot(y_sol_total1, y_sol_total3)
plt.xlabel(r'tetta sw', fontsize=16)
plt.ylabel(r'tetta sw dot', fontsize=16)
plt.show()

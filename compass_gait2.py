import numpy as np
from numpy.linalg import inv as inv
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


#Параметры системы

fi = np.pi/3 #наклон поверхности
mu=1/2 #отношение массы туловища к бедру
m = 5 #масса бедра
m_h = 10 #масса туловища
#сегменты ног
a = 0.5
b = 0.5
bet = 1
l = a+b
g = 9.81# ускорение свободного падения

#считаем число раз смены ноги
flag = 0

def full_energy(y):
	#при ударе теряется энергия
	alpha = (y[1]-y[0])/2
	M = np.array([[bet**2, -(1+bet)*bet*np.cos(2*alpha)],[-(1+bet)*bet*np.cos(2*alpha), (1+bet)**2*(mu+1) +1]])*m*a**2
	q_dot = np.array([y[2],y[3]])
	T = 1/2*((np.transpose(q_dot)).dot(M)).dot(q_dot)
	P = (m*(a+l) + m_h*l)*g*np.cos(2*np.pi - y[1]) - m*g*np.cos(2*np.pi +y[0]-2*y[1])	
	return P+T	

def one_leg_phase(t, y_tmp): 
    #умножаю на матрицу перехода после смены опорной ноги число раз -- flag
    alpha = (y_tmp[1] - y_tmp[0])/2
    Q_minus = np.array([[-bet, -bet+(mu*(1+bet)**2+2*(1+bet))*np.cos(2*alpha)],[0,-bet]])
    Q_plus = np.array([[bet*(bet-(1+bet)*np.cos(2*alpha)), (1+bet)*((1+bet)-bet*np.cos(2*alpha))+1+mu*(1+bet)**2],[bet**2, -bet*(1+bet)*np.cos(2*alpha)]])
    y=y_tmp
    for i in range(0,flag):
        derivation_tmp = ((inv(Q_plus)).dot(Q_minus)).dot([[y[2]],[y[3]]])
        y = np.array([y[1],y[0],float(derivation_tmp[0]),float(derivation_tmp[1])])

    alpha = (y[0]-y[1])/2
    M = np.array([[bet**2, -(1+bet)*bet*np.cos(2*alpha)], [-(1+bet)*bet*np.cos(2*alpha), (1+bet)**2*(mu+1)+1]])
    N = np.array([[0, (1+bet)*bet*y[2]*np.sin(y[0] - y[1])],  [-(1+bet)*bet*y[3]*np.sin(y[0] - y[1]), 0]])
    G_ = np.array([[g*bet*np.sin(y[1])], [-((mu+1)*(1+bet)+1)*g*np.sin(y[0])]])
    tmp_right_side = -((inv(M)).dot(N)).dot(np.array([[y[2]],[y[3]]]))-1/alpha*(inv(M)).dot(G_)

    res = [0,0,0,0]
    res[0] = y[0]
    res[1] = y[1]
    res[2] = float(tmp_right_side[0])
    res[3] = float(tmp_right_side[1])
    return res

def hit_detection(t,y):
	#print(2*fi+y[0]+y[1]-4*np.pi)
	return 2*fi+y[0]+y[1]-4*np.pi
	

y_0 = np.array([np.pi - np.pi/3, np.pi - np.pi/5, -0.8, 2.1])
t_start = 1
t_end = 10
#hit_detection.terminal = True
#hit_detection.direction = 0

ys = [y_0]
ys_1 = [y_0[1]]
ys_3 = [y_0[3]]

while True:
	sol = solve_ivp(one_leg_phase, (t_start, t_end), y_0, events=hit_detection, max_step = 0.1)
	
	ys.append(sol.y[:, -1])
	ys_1.append(sol.y[1, -1])
	ys_3.append(sol.y[3, -1])
	print(sol.t_events)
	if len(sol.t_events[0]) < 1:
		break
	else:
		print('ks')
		flag += 1
	y_0 = sol.y[:, -1]
	t_start = sol.t_events[0]


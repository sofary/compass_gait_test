import numpy as np
from numpy.linalg import inv as inv
import matplotlib.pyplot as plt
import math
from tqdm import *

a = np.array([[0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0],
              [0, 1.0/5, 0, 0, 0, 0, 0, 0],
              [0, 3.0/40, 9.0/40, 0, 0, 0, 0, 0],
              [0, 44.0/45, -56.0/15, 32.0/9, 0, 0, 0, 0],
              [0, 19372.0/6561, -25360.0/2187, 64448.0/6561, -212.0/729, 0, 0, 0],
              [0, 9017.0/3168, -355.0/33, 46732.0/5247, 49.0/176, -5103.0/18656, 0, 0],
              [0, 35.0/384, 0, 500.0/1113, 125.0/192, -2187.0/6784, 11.0/84, 0]])
        #классика y1 (стр 182)     
b = np.array([0, 35.0/384, 0, 500.0/1113, 125.0/192, -2187.0/6784, 11.0/84, 0])
#вычисление для толлерантности и смены шага (182 стр)
b_k = np.array([0, 5179.0/57600, 0, 7571.0/16695, 393.0/640, -92097.0/339200, 187.0/2100, 1.0/40])
c = np.array([0, 0, 1.0/5, 3.0/10, 4.0/5, 8.0/9, 1.0, 1.0])
#параметры система
bet=1
fi = 30
mu=2
g = 9.81
#Порядок метода
P = 5 
eps = 10**(-8)  #Погрешность вычислений

             
#Норма для метода Рунге-Кутта (стр 177)
def Norma(x, y):         
    return np.abs(x - y).max() / (pow(2, P)-1)

#Обычная евклидова норма
def UsualNorma(x):
    return np.sqrt(np.sum(x**2))
    
#Векторное поле
def f(t, x_tmp, res, bet, mu):

    #переключения
    global number_of_steps
    alpha = (x_tmp[1]-x_tmp[0])/2

    if t//0.1>number_of_steps:
        Q_minus = np.array([[-bet, -bet+(mu*(1+bet)**2+2*(1+bet))*np.cos(2*alpha)],[0,-bet]])
        Q_plus = np.array([[bet*(bet-(1+bet)*np.cos(2*alpha)), (1+bet)*((1+bet)-bet*np.cos(2*alpha))+1+mu*(1+bet)**2],[bet**2, -bet*(1+bet)*np.cos(2*alpha)]])
        derivation_tmp = ((inv(Q_plus)).dot(Q_plus)).dot([[x_tmp[3]],[x_tmp[2]]])
        x = np.array([x_tmp[1],x_tmp[0],float(derivation_tmp[0]),float(derivation_tmp[1])])
        number_of_steps+=1
    else:
        x = x_tmp
    
    #Рассчеты матриц
    M = np.array([[bet**2, -(1+bet)*bet*np.cos(2*alpha)],[-(1+bet)*bet*np.cos(2*alpha), (1+bet)**2*(mu+1) +1]]) #inv
    N_matr = np.array([[0, (1+bet)*bet*x[3]*np.sin(x[1]-x[0])],[-(1+bet)*bet*x[2]*np.sin(x[1]-x[0]), 0]])
    g_matr = np.array([[g*bet*np.sin(x[0])], [-((mu+1)*(1+bet)+1)*g*np.sin(x[1])]])
    tmp_right_side = -((inv(M)).dot(N_matr)).dot(np.array([[x[2]],[x[3]]]))-1/alpha*(inv(M)).dot(g_matr)

    res[0] = x[0]
    res[1] = x[1]
    res[2] = tmp_right_side[0]
    res[3] = tmp_right_side[1]
    return True


#RK method
def RungeKutta(h, x_0, t_0, T, N, tol = 10**(-8)):#data=np.array([1, 1]) матрица А y -- b
    r_x = np.zeros((N, 1))
    for i in range(N):
        r_x[i][0] = x_0[i]
    x = np.zeros(N)
    t = t_0

    tmp_x = np.zeros(N)
    kx = np.zeros((8, N)) #коэффициенты k при вычислении сумм в методе Рунге-Кутта
    x_k = np.zeros(N)
    x_p = np.zeros(N) #x_k[N] - значения с крышечкой, x_p[N] - переменные для суммирования

    #параметры для автошага
    fac = 0.9
    facmax = 1.5 
    facmin = 0.7 

    steps = 0 #число шагов
    times = [0]
    t_n = 0
    x = x_0
    
    flag = True
    m_h = [h]
    while t < T:
        #вычисление сумм аргументов
        for i in range(1, 8):
            tmp_p = np.zeros(N)
            tmp_x = [x[l] for l in range(N)]

            for j in range(1, i):
                for l in range(N):
                    tmp_x[l] = tmp_x[l] + h*a[i][j]*kx[j][l]
            f(t + c[i]*h, tmp_x, kx[i], bet, mu)
            # все вычисляется рекуррентно  
        for l in range(N):
            x_p[l] = x[l]
            x_k[l] = x[l]

        for i in range(1, 8): #прибавляем к старым значениям довески с суммами
            for l in range(N):
                x_p[l] += h*b[i]*kx[i][l]#новое значение решения, те результат интегрирования
                x_k[l] += h*b_k[i]*kx[i][l]
        
        #условие на то, допустима ли точность следующего шага
        if Norma(x_p, x_k) < tol:#если погрешность велика, то шаг не делаем
            x = x_p
            r_x = np.append(r_x, x.reshape((N,1)), axis=1)
            #сделать массив с шагами, с учетом того, что не все попытки шага удачные
            t += h
            times.append(t)
            steps += 1#если толерантность меньше, то шаг окей

    #шаг меняется каждый шаг, адаптируемся под кривизну, согласно учебнику 
        h = h * min(facmax, max(facmin, fac * math.pow(tol / Norma(x, x_k), 1.0/(P + 1))))
        m_h.append(h)
        steps -= 1#если толернтность плохая, то делаем перерасчет


    plt.plot(times, r_x[0, :])
    plt.xlabel("time")
    plt.ylabel("Tetta_ns")
    plt.show()
    plt.plot(times, r_x[1, :])
    plt.xlabel("time")
    plt.ylabel("Tetta_s")
    plt.show()
    plt.plot(times, r_x[2, :])
    plt.xlabel("time")
    plt.ylabel("Tetta_dot_ns")
    plt.show()
    plt.plot(times, r_x[3, :])
    plt.xlabel("time")
    plt.ylabel("Tetta_dot_s")
    plt.show()
    plt.plot(r_x[0, :], r_x[2, :])
    plt.xlabel("Tetta_ns")
    plt.ylabel("Tetta_dot_ns")
    plt.show()
    plt.plot(r_x[1, :], r_x[3, :])
    plt.xlabel("Tetta_s")
    plt.ylabel("Tetta_dot_s")
    plt.show()
    plt.plot(np.array(m_h))
    plt.show()
    return times, r_x

h = 0.01
x_0 = np.array([np.pi ,np.pi/4,2,-3])
t_0 = 0
T = 5
N = 4
tol = 10**(-8)
number_of_steps = 0


RungeKutta(h, x_0, t_0, T,N, tol = 10**(-8))
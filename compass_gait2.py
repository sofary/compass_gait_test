import numpy as np
from numpy.linalg import inv as inv
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from celluloid import Camera

#Константы
m = 5
M_big = 15
L = 1
mu = m/M_big
gamma = 0.009 #угол наклона плоскости
g = 9.81

#tetta, fi, tetta_d, fi_d
y_0 = np.array([-(np.pi/40 + 0.01),np.pi/20,0,0])

t_start = 0
t_end = 8

y_sol_total0 = []
y_sol_total1 = []
y_sol_total2 = []
y_sol_total3 = []
interraotion_t = []
hit_t = []

#Интегрируемая система	
def one_leg_phase(t, y): 
    res = [0,0,0,0]
    res[0] = y[2]
    res[1] = y[3]
    res[2] = np.sin(-y[0])
    res[3] = res[2] + (y[2]**2)*np.sin(y[1])-np.cos(-y[0])*np.sin(y[1])
    return res

#событие с геометрическим условием переключения
def hit_detection(t,y):
    # 1 ВАРИАНТ
    #равенство угла фи двум углам тетта
    # func = -2*y[0]-y[1]

    # 2 ВАРИАНТ
    # если стопы ног образуют линию того же наклона, что наша наклонная плоскость
    x1 = L*np.sin(-y[0] - gamma)
    y1 = -L*np.cos(-y[0]-gamma)
    x2 = -L*np.sin(y[1]+y[0]-gamma)
    y2 = -L*np.cos(y[1]+y[0]-gamma)
    func = np.tan(np.pi - gamma) - (y2-y1)/(x2-x1)
    #Результаты 1 и 2 ВАРИАНТОВ одинаковы
    return func

#проверка равенства нулю угла мужду звеньев ног фи
def fi_zero_detection(t,y):
    return y[1]

#tetta, fi, tetta_d, fi_d
#матрица перехода при переключении
def changing_leg(y):
    y_new = [0,0,0,0]
    y_new[0] = -y[0]-y[1]
    y_new[1] = -y[1]
    y_new[2] = y[2]*np.cos(-2*y[0])
    y_new[3] = y[2]*np.cos(-2*y[0])*(1 - np.cos(-2*y[0]))
    return y_new


#цикл интегрирования
while True:
    sol = solve_ivp(one_leg_phase, (t_start, t_end), y_0,   events=(hit_detection,fi_zero_detection), t_eval=np.linspace(t_start, t_end, 100))
    #если не было событий, то выходим и записываем решение
    if len(sol.t_events[0]) < 1:
        for i in range(0,len(sol.y[1])):
            if t_start <= sol.t[i] and sol.t[i] <= t_end:
                y_sol_total0.append(sol.y[0][i])
                y_sol_total1.append(sol.y[1][i])
                y_sol_total2.append(sol.y[2][i])
                y_sol_total3.append(sol.y[3][i])
                interraotion_t.append(sol.t[i])
        break

    indek = 0
    #выбираем событие, чтоб во время переключения угол фи ! = 0
    for elem in sol.t_events[0]:
        if elem not in sol.t_events[1] and indek ==0:
            sol_t_event = elem
            indek = 1
    #если было одно событие и оно совпало с фи = 0, то записываем решение и выходим
    if len(sol.t_events[0])==1 and (sol.t_events[0][0] in sol.t_events[1]):
        for i in range(len(sol.y[1])):
            if t_start <= sol.t[i] and sol.t[i] <= t_end:
                y_sol_total0.append(sol.y[0][i])
                y_sol_total1.append(sol.y[1][i])
                y_sol_total2.append(sol.y[2][i])
                y_sol_total3.append(sol.y[3][i])
                interraotion_t.append(sol.t[i])
        break
        
    #записываем решение
    for i in range(len(sol.y[1])):
        if sol.t[i] <= sol_t_event: 
            y_sol_total0.append(sol.y[0][i])
            y_sol_total1.append(sol.y[1][i])
            y_sol_total2.append(sol.y[2][i])
            y_sol_total3.append(sol.y[3][i])
            interraotion_t.append(sol.t[i])

    #меняем начальное время интегрирования
    t_start = sol_t_event
    hit_t.append(t_start)
    #меняем опорную и переносную ноги местами
    y_0 =  changing_leg(y_0)



#Графики

plt.plot(interraotion_t, y_sol_total0,'orange', interraotion_t, y_sol_total1,'b')
plt.legend(('tetta','fi'))
plt.ylabel("angle, radians", fontsize=15)


fig2, axs = plt.subplots(nrows= 1 , ncols= 2 )
fig2. suptitle('tetta dot (tetta)                  /                fi dot (fi)')
axs[0].plot(y_sol_total2,y_sol_total0)
axs[1].plot(y_sol_total3, y_sol_total1)
axs[0].set_ylabel(" angular velocity, d radians/dt", fontsize=10)
axs[0].set_xlabel(" angular, radians", fontsize=10)
axs[1].set_xlabel(" angular, radians", fontsize=10)


fig3, axs = plt.subplots(nrows= 1 , ncols= 2 )
fig3. suptitle('d tetta                   /                     d fi')
axs[0].plot(interraotion_t, y_sol_total2)
axs[1].plot(interraotion_t, y_sol_total3)
axs[0].set_ylabel("angular velocity, d radians/dt", fontsize=10)
plt.show()


##################################################
print('Tetta_angl =',y_sol_total0,';')
print('----')
print('fi_angl = ',y_sol_total1,';')
print('----')
print('interraotion_t = ',interraotion_t,';')#[27:116]
print('----')
print('hit_t = ',hit_t,';')
##################################################

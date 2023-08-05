import numpy as np
from numpy.linalg import inv as inv
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from celluloid import Camera


#Параметры системы
m = 5
M_big = 10
a = 0.5
b = 0.5
l = a + b
bet = m/M_big
mu = a/l
v = b/l 
alpha = np.pi/12
g = 9.81
flag = 0
	
def one_leg_phase(t, y_tmp): 
    res = [0,0,0,0]
    y = y_tmp
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
    return res

def hit_detection(t,y):
    return 2*alpha+y[0]+y[1]-4*np.pi

#y_0 = np.array([np.pi*2 - np.pi/6, np.pi*2 - np.pi/6, 0.01, 0.01])
#y_0 = np.array([ -310/180*np.pi, 140/180*np.pi, 0, 0])
y_0 = np.array([ np.pi*2 - np.pi/15 , np.pi*2 - np.pi/10, 0, 0])
t_start = 0.01
t_end = 6


y_sol_total0 = [y_0[0]]
y_sol_total1 = [y_0[1]]
y_sol_total2 = [y_0[2]]
y_sol_total3 = [y_0[3]]
interraotion_t = [0]
hit_t = []

while True:
    print(y_0)
    sol = solve_ivp(one_leg_phase, (t_start, t_end), y_0,   events=hit_detection)#, t_eval=np.linspace(t_start, t_end, 10))#, min_step = 0.01)#, t_eval=np.linspace(t_start, t_end, 100))
    if len(sol.t_events[0]) < 1:
        for i in range(0,len(sol.y[1])):
            if t_start <= sol.t[i] and sol.t[i] <= t_end:
                if flag % 2 ==0:
                    y_sol_total0.append(sol.y[0][i])
                    y_sol_total1.append(sol.y[1][i])
                    y_sol_total2.append(sol.y[2][i])
                    y_sol_total3.append(sol.y[3][i])
                else:
                    y_sol_total0.append(sol.y[1][i])
                    y_sol_total1.append(sol.y[0][i])
                    y_sol_total2.append(sol.y[3][i])
                    y_sol_total3.append(sol.y[2][i])
                interraotion_t.append(sol.t[i])
        break
    #выбираем событие
    if len(sol.t_events[0])>1 and sol.t_events[0][0]==0.01:
        sol_t_event = sol.t_events[0][1]
    else:
        sol_t_event = sol.t_events[0][0]
    for i in range(0, len(sol.y[1])):
        if sol.t[i] <= sol_t_event: 
            if flag % 2 ==0:
                y_sol_total0.append(sol.y[0][i])
                y_sol_total1.append(sol.y[1][i])
                y_sol_total2.append(sol.y[2][i])
                y_sol_total3.append(sol.y[3][i])
            else:
                y_sol_total0.append(sol.y[1][i])
                y_sol_total1.append(sol.y[0][i])
                y_sol_total2.append(sol.y[3][i])
                y_sol_total3.append(sol.y[2][i])
            interraotion_t.append(sol.t[i])

    
    if abs(t_start - sol_t_event) < 0.01 :#or abs(y_sol_total3[-1])>5 or abs(y_sol_total2[-1])>5:
        print("----")
        break
    t_start = sol_t_event
    hit_t.append(t_start)
    if flag % 2 == 0:
        y_0 = np.array([y_sol_total0[-1], y_sol_total1[-1], y_sol_total2[-1], y_sol_total3[-1]])
    else:
        y_0 = np.array([y_sol_total1[-1], y_sol_total0[-1], y_sol_total3[-1], y_sol_total2[-1]])
    flag += 1
    #преобразование координат при смене ноги

    fi = y_0[1]-y_0[0]
    v_minus = np.array([[-bet*v*mu + (2*bet*v+1)*np.cos(fi), -bet*v*mu],[-bet*v*mu, 0]])
    v_plus = np.array([[bet*v**2 + 1 + bet - bet*mu*np.cos(fi), bet*v**2 - bet*v*np.cos(fi)],[-bet*v*np.cos(fi), bet*v**2]])
    K = np.concatenate([[[0,0],[0,0]],np.dot(inv(v_plus), v_minus)], axis = 1)
    Up_part = np.array([[0,1,0,0],[1,0,0,0]])
    J = np.concatenate([Up_part, K])
    y_0 =  np.dot(J,y_0)


print(len(interraotion_t))
print(len(hit_t))
h=1
#Графики
fig1, axs = plt.subplots(nrows= 1 , ncols= 2 )
fig1. suptitle('tetta st                  /                    tetta sw')
axs[0].plot(interraotion_t, y_sol_total0)
axs[1].plot(interraotion_t, y_sol_total1)


fig2, axs = plt.subplots(nrows= 1 , ncols= 2 )
fig2. suptitle('tetta st dot (tetta st)                  /             tetta sw dot (tetta sw)')
axs[0].plot(y_sol_total0, y_sol_total2)
axs[1].plot(y_sol_total1, y_sol_total3)

fig3, axs = plt.subplots(nrows= 1 , ncols= 2 )
fig3. suptitle('d tetta st                  /                     d tetta sw')
axs[0].plot(interraotion_t, y_sol_total2)
axs[1].plot(interraotion_t, y_sol_total3)
plt.show()
##################################################

t0 = 20
x0 = 20
t = np.linspace(0, 100, 100)


fig = plt.figure()
 
camera = Camera(fig)
for i in range(len(y_sol_total0)):
    shift0 = x0 - t0*np.tan(np.pi - y_sol_total0[i])
    shift1 = x0 - t0*np.tan(np.pi - y_sol_total1[i])
    pp = plt.plot(t, t*np.tan(np.pi - alpha),'r--', t, t*np.tan(np.pi - y_sol_total0[i])+shift0,'b-', t, t*np.tan(np.pi - y_sol_total1[i]) + shift1,'g-')
    plt.xlim([0, 80])
    plt.ylim([-80, 80])
    plt.legend(pp, ['number of iteration{}'.format(i)])
    camera.snap()
 
animation = camera.animate()
animation.save('walk.gif', writer = 'imagemagick')

#########################################################

# for i in range(len(y_sol_total0)):
#     shift0 = x0 - t0*np.tan(np.pi - y_sol_total0[i])
#     shift1 = x0 - t0*np.tan(np.pi - y_sol_total1[i])
#     plt.figure(i)
#     plt.plot(t, t*np.tan(np.pi - alpha),'r--', t, t*np.tan(np.pi - y_sol_total0[i])+shift0,'b-', t, t*np.tan(np.pi - y_sol_total1[i]) + shift1,'g-')
#     plt.xlim([0, 80])
#     plt.ylim([-80, 80])
#     plt.savefig('graph{}.png'.format(i))
#plt.show()

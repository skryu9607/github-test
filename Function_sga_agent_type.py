import random
import time
import math
import copy
import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt

# Agent Type
agent_type_list = ["Surv","Combat","Mine"]
# Agent_vel = '30km/h', '30km/h', '10km/h' (이동) 3.6을 나누면 m/s
# Agent_vel = '4km/h', '2km/h', '2km/h' (이동감시, 화학탐지) 2,3
# Agent_vel =  0, 0, 5km/h (지뢰탐지) 4
agent_vel_mat = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], # (고정)
                 [4.0/3.6/5, 2.0/3.6/5, 2.0/3.6/5], [4.0/3.6/5, 2.0/3.6/5, 2.0/3.6/5], # (이동감시, 화학탐지)
                 [0.0, 0.0, 5.0/3.6/5], # (지뢰탐지)
                 [30.0/3.6/10, 30.0/3.6/10, 10.0/3.6/10]] # (이동)
# Task Type
task_type_list = ['attack_wait', 'fix_surv', 'mov_surv', 'chem', 'mine']

# Reward
reward_mat = [[0.8, 1.0, 0.2], [1.0, 0.8, 0.5], [1.0, 0.8, 0.01], [1.0, 1.0, 0.5], [0.01, 0.01, 1.0]]

class Info:
    def __init__(self, agents, tasks):
        self.Agents = agents
        self.Tasks = tasks
        self.task_pos = []
        self.task_type = []
        for i in range(len(tasks)):
            self.task_pos.append([tasks[i].x1,tasks[i].y1,tasks[i].x2,tasks[i].y2])
            self.task_type.append(tasks[i].type)

class Agent:
    def __init__(self, id, agent_type, x, y):
        self.id = id
        self.type = agent_type
        self.x = x
        self.y = y
        self.bundle = []
        self.path = []

class Task:
    def __init__(self, id, task_type, x1, y1):
        self.id = id
        self.type = task_type
        self.x1 = x1
        self.y1 = y1
        if task_type > 1:
            self.x2 = x1 + random.random() - 0.5
            self.y2 = y1 + random.random() - 0.5
        else:
            self.x2 = x1
            self.y2 = y1
        self.start = -1
        self.end = -1
        self.agent = -1
        self.duration = -1
        if task_type == 0:
            self.duration = 10000
        self.reward = 1000
        self.constraints_o = []
        self.constraints_n = []

def Initialize():
    agents = []
    # for i in range(6):
    #     tp1 = random.randint(0,2)
    #     tp2 = random.random()
    #     tp3 = random.random()
    #     agents.append(Agent(i,tp1,tp2,tp3))
    #     print(i,",",tp1,",",tp2,",",tp3)
    agents.append(Agent(0 , 1 , 0.6126933948977238 , 0.8064742156252204))
    agents.append(Agent(1 , 1 , 0.2331169528374767 , 0.02413844273592658))
    agents.append(Agent(2 , 0 , 0.6229014668676646 , 0.16778732377394823))
    agents.append(Agent(3 , 1 , 0.35238038752798173 , 0.5514289495977904))
    agents.append(Agent(4 , 0 , 0.03425606533963699 , 0.06558967378898561))
    agents.append(Agent(5 , 2 , 0.9763099467011696 , 0.4691696215075748))

    tasks = []
    task_type = []
    # for i in range(50):
    #     if task_type.count(0) > 5:
    #         task_type.append(random.randint(1,4))
    #     else:
    #         task_type.append(random.randint(0,4))
    #     tp1 = random.random()*8+1.5
    #     tp2 = random.random()*8+1.5
    #     tasks.append(Task(i,task_type[i],tp1,tp2))
    #     print(i,",",task_type[i],",",tp1,",",tp2)
    tasks.append(Task(0 , 2 , 4.352920911940889 , 6.979938186321736))
    tasks.append(Task(1 , 1 , 6.897927079065055 , 3.44864981175869))
    tasks.append(Task(2 , 1 , 6.416902194737278 , 5.638226466563731))
    tasks.append(Task(3 , 1 , 6.750894209452262 , 3.5983013135762247))
    tasks.append(Task(4 , 1 , 6.076919416719013 , 3.007987866745907))
    tasks.append(Task(5 , 4 , 7.557823476521551 , 3.7491961154776243))
    tasks.append(Task(6 , 1 , 7.71502955247061 , 2.27435224509757))
    tasks.append(Task( 7 , 2 , 8.672827519919938 , 3.349970447457035))
    tasks.append(Task( 8 , 2 , 3.058757329162736 , 8.375365739235072))
    tasks.append(Task(9 , 2 , 5.924246498412136 , 7.262648871978236))
    tasks.append(Task(10 , 2 , 6.224368853024192 , 3.585315815151211))
    tasks.append(Task( 11 , 3 , 2.5995230776309093 , 5.094538989034338))
    tasks.append(Task(12 , 2 , 3.170145104241537 , 3.2546450440146293))
    tasks.append(Task(13 , 4 , 8.342845656919609 , 3.190667598746119))
    tasks.append(Task(14 , 2 , 1.812052110810784 , 6.233807962159626))
    tasks.append(Task(  15 , 3 , 3.72987912830202 , 7.9170735290206204))
    tasks.append(Task(  16 , 1 , 7.172972391159184 , 7.868949445794964))
    tasks.append(Task( 17 , 3 , 7.718884775361162 , 7.889956257721879))
    tasks.append(Task( 18 , 2 , 2.1997247347909035 , 6.123818002064363))
    tasks.append(Task(19 , 2 , 5.706406951206918 , 2.895666131762714))
    tasks.append(Task(20 , 4 , 8.093052904057956 , 6.289203259749064))
    tasks.append(Task(21 , 4 , 5.982657629772439 , 4.613199433539062))
    tasks.append(Task(22 , 1 , 5.97752475984492 , 6.57487579628337))
    tasks.append(Task(23 , 1 , 5.494801203420019 , 2.1637713576637294))
    tasks.append(Task(24 , 2 , 8.551766665567369 , 3.570333003697737))
    tasks.append(Task(25 , 2 , 5.443243696725784 , 6.235110501162521))
    tasks.append(Task(26 , 1 , 2.0011361778462113 , 8.397907184454557))
    tasks.append(Task(27 , 4 , 6.467457983004264 , 4.585424866606174))
    tasks.append(Task(28 , 2 , 5.349263884860281 , 7.38507105810634))
    tasks.append(Task(29 , 1 , 7.534917450142256 , 7.52478793652159))
    tasks.append(Task(30 , 1 , 4.728148758864316 , 5.368704865478125))
    tasks.append(Task(31 , 3 , 4.280535843045456 , 5.0507734840187934))
    tasks.append(Task(32 , 2 , 2.9391074212881003 , 4.572250328996229))
    tasks.append(Task(33 , 3 , 9.315783674557437 , 8.035301614202687))
    tasks.append(Task(34 , 3 , 4.986321117640675 , 6.865810527027347))
    tasks.append(Task(35 , 1 , 4.5205199873703386 , 6.085531621947713))
    tasks.append(Task(36 , 2 , 7.760078165323966 , 5.3629916445949535))
    tasks.append(Task(37 , 1 , 5.314986989631317 , 2.488439884044978))
    tasks.append(Task(38 , 1 , 1.659154536210413 , 5.24791735905853))
    tasks.append(Task(39 , 3 , 6.901731110487137 , 2.0065591963690492))
    tasks.append(Task(40 , 2 , 1.8790650887033333 , 8.716443844521098))
    tasks.append(Task(41 , 1 , 6.5715770468342685 , 2.2647508896490764))
    tasks.append(Task(42 , 1 , 1.692181475634678 , 4.596475902156895))
    tasks.append(Task(43 , 1 , 7.626169657373432 , 2.835327460250931))
    tasks.append(Task(44 , 4 , 8.443890804091616 , 7.012499958529867))
    tasks.append(Task(45 , 4 , 3.9573184017899754 , 4.720653415676596))
    tasks.append(Task(46 , 2 , 2.190314507453973 , 8.737262573858295))
    tasks.append(Task(47 , 2 , 4.234304205356387 , 8.138502330972269))
    tasks.append(Task(48 , 2 , 4.230552163112075 , 7.810300735828707))
    tasks.append(Task(49 , 1 , 6.200793969615892 , 2.563393712038712))

    info = Info(agents, tasks)

    return info

def Cal_Reward(info, agent_id, path):
    reward = 0
    agent_type = info.Agents[agent_id].type
    task_type = info.task_type
    for i in range(len(path)):
        reward += 1000 * reward_mat[task_type[path[i]]][agent_type]
    return reward

def Cal_Time(info, agent_id, path):
    t = 0
    agent_x = info.Agents[agent_id].x
    agent_y = info.Agents[agent_id].y
    agent_type = info.Agents[agent_id].type
    agent_vel = agent_vel_mat[5][agent_type]
    task_type = info.task_type
    task_pos = info.task_pos
    for i in range(len(path)):
        x1 = task_pos[path[i]][0]
        y1 = task_pos[path[i]][1]
        x2 = task_pos[path[i]][2]
        y2 = task_pos[path[i]][3]
        if i == 0:
            t += (np.sqrt((x1 - agent_x)**2 + (y1 - agent_y)**2)) / agent_vel
        # 거리 계산 : 그 이후에는 Path i 번과 i-1 번 간 거리 측정
        else:
            x2_p = task_pos[path[i-1]][2]
            y2_p = task_pos[path[i-1]][3]
            t += np.sqrt((x1 - x2_p)**2 + (y1 - y2_p)**2) / agent_vel
        # 임무 수행하는 거리
        dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
        if dist == 0:
            if task_type[path[i]] == 0:
                t += 100
            elif task_type[path[i]] == 1:
                t += 10
            else:
                print("task_pos Error in Cal_Time")
        else:
            t += min(100, dist / max(agent_vel_mat[task_type[path[i]]][agent_type],0.1))

    if len(path) > 0:
        if not path[-1] == 0:
            x2_e = task_pos[path[-1]][2]
            y2_e = task_pos[path[-1]][3]
            t += np.sqrt((x2_e - 0)**2 + (y2_e - 0)**2) / agent_vel

    return t

def Cal_Score(info, agent_id, path):
    rr = Cal_Reward(info, agent_id, path)
    tt = Cal_Time(info, agent_id, path)
    if tt > 100:
        return 10
    else:
        return rr - tt

def Cal_Max_Score(Info, agent_id, new_task):
    Score = []
    agent = Info.Agents[agent_id]
    if len(agent.path) == 0:
        Score.append(Cal_Score(Info, agent_id, [new_task]))
    else:
        for i in range(len(agent.path) + 1):
            Path = copy.deepcopy(agent.path)
            Path.insert(i, new_task)
            Score.append(Cal_Score(Info, agent_id, Path))
    arg = np.argmax(Score)
    return max(Score), arg
def gen_mat(Path, new_task, pos):
    New_Path = []
    cnt = 0
    for i in range(len(Path)+1):
        if i == pos:
            New_Path.append(new_task)
        else:
            New_Path.append(Path[cnt])
            cnt = cnt + 1
    return New_Path
def SGA(info):
    # Initialize CBBA parameters
    agents = info.Agents
    tasks = info.Tasks
    N_agent = len(agents)
    N_task = len(tasks)
    MAX_LENGTH = int(len(tasks)/2)
    llambda = 0.1

    cost_mat = np.zeros((N_agent, N_task))
    agent_vec = np.arange(0, N_agent, dtype=int)
    task_vec = np.arange(0, N_task, dtype=int)
    agent_vec = agent_vec.tolist()
    task_vec = task_vec.tolist()
    eta = np.zeros(N_agent)

    for i in range(N_agent):
        for j in range(N_task):
            cost_mat[i,j] = tasks[j].reward - LA.norm([agents[i].x - tasks[j].x1, agents[i].y - tasks[j].y1])

    for cnt in range(N_task):

        info.agent_vec = agent_vec
        info.task_vec = task_vec
        cost_mat_ = np.zeros((N_agent, N_task))
        for i in range(N_agent):
            if i in agent_vec:
                for j in range(N_task):
                    if j in task_vec:
                        cost_mat_[i, j] = cost_mat[i, j]

        argmax_cost = np.argmax(cost_mat_)
        # N_agent 가 행의 갯수, N_task 가 열의 갯수
        # cost_mat 의 i, j 성분은 i * N_task + j 이라는 숫자로 나타난다
        argmax_agent = int(argmax_cost/N_task)
        argmax_task = argmax_cost - argmax_agent * N_task
        eta[argmax_agent] += 1
        # print(cnt)
        if len(task_vec) > 0:
            task_vec.pop(np.where(task_vec == argmax_task)[0][0])
        # bundle 생성
        agents[argmax_agent].bundle.append(argmax_task)
        # path 생성
        max_val, ind = Cal_Max_Score(info, argmax_agent, argmax_task)
        agents[argmax_agent].path.insert(ind,argmax_task)
        tasks[argmax_task].agent = argmax_agent

        if eta[argmax_agent] == MAX_LENGTH:
            agent_vec.pop(argmax_agent)
            for j in range(N_task):
                if j in task_vec:
                    cost_mat[argmax_agent,j] = 0
        for i in range(N_agent):
            if i in agent_vec:
                cost_mat[i,argmax_task] = 0

        info.Agents = agents
        info.Tasks = tasks
        for i in range(N_agent):
            if i in agent_vec:
                for j in range(N_task):
                    if j in task_vec:
                        if len(agents[i].bundle) > 0:
                            max_val, ind = Cal_Max_Score(info, i, j)
                            if max_val < Cal_Score(info, i, agents[i].path):
                                cost_mat[i, j] = 1
                            else:
                                # print(i,j,max_val)
                                cost_mat[i, j] = (max_val - Cal_Score(info, i, agents[i].path)) \
                                                 * math.exp(
                                    -llambda * Cal_Time(info, i, gen_mat(agents[i].path, j, ind)))
                                # print(i,j,Cal_Time(info, i, gen_mat(agents[i].path, j, ind)),cost_mat[i, j] )

    for i in range(6):
        for j in range(len(agents[i].path)):
            tasks[agents[i].path[j]].agent = i
    info.Agents = agents
    info.Tasks = tasks
    return info

def Cal_Time_by_taskID(Info, taskid):
    agent_id = Info.Tasks[taskid].agent
    if agent_id == -1:
        return 0
    else:
        t = 0
        agent_x = info.Agents[agent_id].x
        agent_y = info.Agents[agent_id].y
        agent_type = info.Agents[agent_id].type
        agent_vel = agent_vel_mat[5][agent_type]
        task_type = info.task_type
        task_pos = info.task_pos
        path = Info.Agents[agent_id].path
        for i in range(len(path)):
            x1 = task_pos[path[i]][0]
            y1 = task_pos[path[i]][1]
            x2 = task_pos[path[i]][2]
            y2 = task_pos[path[i]][3]
            if i == 0:
                t += (np.sqrt((x1 - agent_x) ** 2 + (y1 - agent_y) ** 2)) / agent_vel
            # 거리 계산 : 그 이후에는 Path i 번과 i-1 번 간 거리 측정
            else:
                x2_p = task_pos[path[i - 1]][2]
                y2_p = task_pos[path[i - 1]][3]
                t += np.sqrt((x1 - x2_p) ** 2 + (y1 - y2_p) ** 2) / agent_vel
            if taskid == path[i]:
                break
            # 임무 수행하는 거리
            dist = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
            if dist == 0:
                if task_type[path[i]] == 0:
                    t += 100
                elif task_type[path[i]] == 1:
                    t += 10
                else:
                    print("task_pos Error in Cal_Time")
            else:
                t += min(100, dist / max(agent_vel_mat[task_type[path[i]]][agent_type], 0.1))
        return t

def Cal_Time_by_taskID2(Info, taskid):
    agent_id = Info.Tasks[taskid].agent
    if agent_id == -1:
        return 0
    else:
        t = 0
        agent_x = info.Agents[agent_id].x
        agent_y = info.Agents[agent_id].y
        agent_type = info.Agents[agent_id].type
        agent_vel = agent_vel_mat[5][agent_type]
        task_type = info.task_type
        task_pos = info.task_pos
        path = Info.Agents[agent_id].path
        for i in range(len(path)):
            x1 = task_pos[path[i]][0]
            y1 = task_pos[path[i]][1]
            x2 = task_pos[path[i]][2]
            y2 = task_pos[path[i]][3]
            if i == 0:
                t += (np.sqrt((x1 - agent_x) ** 2 + (y1 - agent_y) ** 2)) / agent_vel
            # 거리 계산 : 그 이후에는 Path i 번과 i-1 번 간 거리 측정
            else:
                x2_p = task_pos[path[i - 1]][2]
                y2_p = task_pos[path[i - 1]][3]
                t += np.sqrt((x1 - x2_p) ** 2 + (y1 - y2_p) ** 2) / agent_vel

            # 임무 수행하는 거리
            dist = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
            if dist == 0:
                if task_type[path[i]] == 0:
                    t += 100
                elif task_type[path[i]] == 1:
                    t += 10
                else:
                    print("task_pos Error in Cal_Time")
            else:
                t += min(100, dist / max(agent_vel_mat[task_type[path[i]]][agent_type], 0.1))
            if taskid == path[i]:
                break
        return t

def Record_Time(info):
    for i in range(len(info.Tasks)):
        info.Tasks[i].start = Cal_Time_by_taskID(info, info.Tasks[i].id)
        info.Tasks[i].end = Cal_Time_by_taskID2(info, info.Tasks[i].id)
        info.Tasks[i].duration = info.Tasks[i].end - info.Tasks[i].start
    return info

def Drawing(info):
    agents = info.Agents
    tasks = info.Tasks
    N_agent = len(agents)
    N_task = len(tasks)
    Draw_path_x = []
    Draw_path_y = []
    for i in range(N_agent):
        Draw_path_x.append([])
        Draw_path_y.append([])
        Draw_path_x[i].append(agents[i].x)
        Draw_path_y[i].append(agents[i].y)
        for j in range(len(agents[i].path)):
            Draw_path_x[i].append(tasks[agents[i].path[j]].x1)
            Draw_path_y[i].append(tasks[agents[i].path[j]].y1)
            Draw_path_x[i].append(tasks[agents[i].path[j]].x2)
            Draw_path_y[i].append(tasks[agents[i].path[j]].y2)
        if not info.Tasks[agents[i].path[-1]].type == 0:
            Draw_path_x[i].append(0)
            Draw_path_y[i].append(0)


    Draw_time_start = []
    Draw_time_end = []
    Draw_total_end = []
    for i in range(N_agent):
        Draw_time_start.append([])
        Draw_time_end.append([])
        for j in range(len(agents[i].path)):
            Draw_time_start[i].append(tasks[agents[i].path[j]].start)
            Draw_time_end[i].append(tasks[agents[i].path[j]].end)

        if info.Tasks[agents[i].path[-1]].type == 0:
            Draw_total_end.append(Draw_time_end[i][-1])
        else:
            Draw_total_end.append(Cal_Time(info,i,agents[i].path))

    fig = plt.figure()
    ax = fig.add_subplot(121, autoscale_on=False, xlim=(0, 10), ylim=(0, 10))
    ax.set_aspect('equal')
    ax.set_aspect('equal')
    ax.grid()
    col = ['red', 'orange', 'yellow', 'green', 'blue', 'purple']
    for i in range(N_task):
        dot, = ax.plot([tasks[i].x1, tasks[i].x2], [tasks[i].y1, tasks[i].y2], 'o', lw=2)
        time_text = plt.text(tasks[i].x1, tasks[i].y1, i)
    for i in range(N_agent):
        dot, = ax.plot(agents[i].x, agents[i].y, 'o', lw=2)
        line = ax.plot(Draw_path_x[i], Draw_path_y[i], 'o-', lw=2, color=col[i])

    ax2 = fig.add_subplot(122)
    for i in range(N_agent):
        plt.text(0, 2*i, str(agent_type_list[agents[i].type]))
        line = ax2.plot([0, Draw_total_end[i]], [2 * i, 2 * i], 'k')
        for j in range(len(agents[i].path)):
            x1 = Draw_time_start[i][j]
            x2 = Draw_time_end[i][j]
            ax2.fill([x1, x1, x2, x2], [2 * i, 2 * i + 1, 2 * i + 1, 2 * i], color=col[i])
            plt.text(x1, 2 * i, agents[i].path[j])
            plt.text(x1, 2 * i-0.1, task_type_list[info.Tasks[agents[i].path[j]].type])
    plt.show()
    return 0

info = Initialize()
tic = time.time()
info = SGA(info) # --> 여기서 과업별 이동시간까지 같이 나옴
info = Record_Time(info)
toc = time.time()
for i in range(len(info.Agents)):
    print(Cal_Reward(info, i, info.Agents[i].path))
    print(agent_type_list[info.Agents[i].type])
Drawing(info)
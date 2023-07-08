import random
import torch
from torch.optim import Adam
from torch import nn

'''修改DDQN为多目标DDQN'''
class DDQN:
    def __init__(self, learning_rate, tor, tb_writer, device):
        self.device = device
        self.model = DQN_net().to(device)  # Q现实
        self.model_optim = Adam(self.model.parameters(), lr=learning_rate)
        # self.scheduler = torch.optim.lr_scheduler.StepLR(self.model_optim, step_size=80, gamma=0.1)
        self.target_model = DQN_net().to(device)  # Q目标
        # self.target_model.Parameters.requires_grad=False
        self.target_model.load_state_dict(self.model.state_dict())
        self.tor = tor
        self.tb_writer = tb_writer
        data = torch.rand(32, 7).to(device)
        tb_writer.add_graph(self.model, input_to_model=(data,))
        tb_writer.add_graph(self.target_model, input_to_model=(data,))

    def net_weights(self, step_num):
        for name, param in self.model.named_parameters():
            self.tb_writer.add_histogram(tag=name, values=param, global_step=step_num)
            self.tb_writer.add_histogram(tag=name + '_grad', values=param.grad, global_step=step_num)
        # for name,param in self.target_model.named_parameters():
        #     self.tb_writer.add_histogram(tag='target---'+name,values=param,global_step=step_num)

    def act(self, state, epsilon):
        if random.random() > epsilon:
            state = self.model.forward(state)
            action = torch.argmax(state).item()
        else:
            action = random.randrange(6)  # 随机执行一个规则
        return action

    def learning(self, batch_size, gamma, memory, step_num):
        s0, a, r, s1 = memory.sample(batch_size)
        q_values = self.model(s0).to(self.device)
        next_q_values = self.model(s1).to(self.device)
        next_q_state_values = self.target_model(s1).to(self.device)

        q_value = q_values.gather(1, a.unsqueeze(1)).squeeze(1)
        next_q_value = next_q_state_values.gather(1, next_q_values.max(1)[1].unsqueeze(1)).squeeze(1)
        expected_q_value = r + gamma * next_q_value
        # Notice that detach the expected_q_value
        loss = (q_value - expected_q_value.detach()).pow(2).mean()  # batch的loss平均值

        self.model_optim.zero_grad()
        loss.backward()
        self.model_optim.step()

        Q_par = self.model.state_dict()
        target_Q_par = self.target_model.state_dict()
        for key in Q_par:
            target_Q_par[key] = Q_par[key] * self.tor + target_Q_par[key] * (1 - self.tor)

        self.target_model.load_state_dict(target_Q_par)
        # if step_num%1000==0:
        #     self.target_model.load_state_dict(Q_par)
        return loss  # loss.item()

    def load_weights(self, model_path):
        if model_path is None: return
        self.model.load_state_dict(torch.load(model_path))

    def save_model(self):
        torch.save(self.model.state_dict(), 'soft_target_model.pkl')


# 定义DDQN的网络结构
class DQN_net(nn.Module):
    def __init__(self, input=7, output=6, hidden=30):
        super(DQN_net, self).__init__()
        self.input_layer = nn.Linear(input, hidden)
        self.relu = nn.ReLU()
        self.hidden_layer1 = nn.Linear(hidden, hidden)
        self.hidden_layer2 = nn.Linear(hidden, hidden)
        self.hidden_layer3 = nn.Linear(hidden, hidden)
        self.hidden_layer4 = nn.Linear(hidden, hidden)
        self.hidden_layer5 = nn.Linear(hidden, hidden)
        self.output_layer = nn.Linear(hidden, output)
        self.sf = nn.Softmax(dim=0)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.relu(x)
        x = self.hidden_layer1(x)
        x = self.relu(x)
        x = self.hidden_layer2(x)
        x = self.relu(x)
        x = self.hidden_layer3(x)
        x = self.relu(x)
        x = self.hidden_layer4(x)
        x = self.relu(x)
        x = self.hidden_layer5(x)
        x = self.relu(x)
        x = self.output_layer(x)
        x = self.sf(x)
        return x

# D=[]  #记忆池
# N=1000   #容量
# layers = 5
# input = 7
# output = 6
# hidden = 30

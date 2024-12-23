import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt  # 导入matplotlib绘图

# 定义超参数
BATCH_SIZE = 64
GAMMA = 0.99
LEARNING_RATE = 0.0001
EPISODES = 200
PRINT_INTERVAL = 10  # 每隔多少步打印一次

# 检查设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# 数据集定义
class SINRDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)

        # 提取数据
        self.states = np.array(self.data.iloc[:, 0].apply(eval).tolist())
        self.actions = np.array(self.data.iloc[:, 1].apply(eval).tolist())
        self.next_states = np.array(self.data.iloc[:, 2].apply(eval).tolist())
        self.rewards = np.array(self.data.iloc[:, 3].tolist())

        # 归一化处理
        self.state_scaler = MinMaxScaler()
        self.reward_scaler = MinMaxScaler()

        self.states = self.state_scaler.fit_transform(self.states)
        self.next_states = self.state_scaler.transform(self.next_states)
        self.rewards = self.reward_scaler.fit_transform(self.rewards.reshape(-1, 1)).flatten()

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        state = torch.tensor(self.states[idx], dtype=torch.float32)
        action = torch.tensor(self.actions[idx], dtype=torch.float32)
        next_state = torch.tensor(self.next_states[idx], dtype=torch.float32)
        reward = torch.tensor(self.rewards[idx], dtype=torch.float32)
        return state, action, next_state, reward


# Q网络定义
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# 训练过程
def train_dqn(dataset, episodes, batch_size, gamma, learning_rate):
    state_dim = dataset[0][0].shape[0]
    action_dim = dataset[0][1].shape[0]

    q_network = QNetwork(state_dim, action_dim).to(device)
    optimizer = optim.Adam(q_network.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
    criterion = nn.MSELoss()


    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    loss_list = []  # 用于记录每个Episode的Loss值

    for episode in range(episodes):
        total_loss = 0
        step = 0
        for state, action, next_state, reward in dataloader:
            state, action, next_state, reward = state.to(device), action.to(device), next_state.to(device), reward.to(
                device)

            q_values = q_network(state)
            with torch.no_grad():
                next_q_values = q_network(next_state).max(1)[0]  # 最大Q值
                target_q_values = reward + gamma * next_q_values

            current_q_values = q_values.gather(1, action.argmax(dim=1, keepdim=True).long())
            loss = criterion(current_q_values, target_q_values.unsqueeze(1))

            optimizer.zero_grad()
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(q_network.parameters(), max_norm=1.0)

            optimizer.step()


            total_loss += loss.item()
            step += 1

            if step % PRINT_INTERVAL == 0:
                print(f"Episode {episode + 1}/{episodes}, Step {step}, Loss: {loss.item():.4f}")

        scheduler.step()  # 调整学习率
        avg_loss = total_loss / step  # 每个Episode的平均Loss
        loss_list.append(avg_loss)
        print(f"Episode {episode + 1} finished, Total Loss: {avg_loss:.4f}")

    # 绘制损失曲线
    plt.plot(range(1, episodes + 1), loss_list)
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.title('Loss Curve during Training')
    plt.show()

    return q_network


# 主程序
if __name__ == "__main__":
    dataset = SINRDataset("DQN.csv")  # 替换为您的CSV文件路径
    trained_model = train_dqn(dataset, EPISODES, BATCH_SIZE, GAMMA, LEARNING_RATE)

    # 保存模型
    torch.save(trained_model.state_dict(), "dqn_model2.pth")
    print("Model saved as dqn_model.pth")

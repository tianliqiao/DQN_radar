import socket
import pickle
import struct
import numpy as np
import matplotlib.pyplot as plt


ip = "192.168.1.7"
port = 5000

# 雷达信号参数
radar_params = [3000000000.0, 17, 0, 5000000.0, 20000.0, 'NegativeTriangle', 4000000000]

# 创建 TCP 连接
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect((ip, port))

try:
    # 序列化数据
    data_to_send = pickle.dumps(radar_params)
    data_length = len(data_to_send)

    # 先发送数据长度
    sock.sendall(struct.pack(">I", data_length))
    # 再发送实际数据
    sock.sendall(data_to_send)
    print("Data sent with length:", data_length)

    # 接收返回数据
    length_data = sock.recv(4)
    if not length_data:
        raise ValueError("No length data received.")
    response_length = struct.unpack(">I", length_data)[0]
    print("Expected response length:", response_length)

    received_data = b""
    while len(received_data) < response_length:
        chunk = sock.recv(4096)
        received_data += chunk
    print("Received data length:", len(received_data))

    radar_waveform = pickle.loads(received_data)
    print("Received radar waveform:", radar_waveform)

finally:
    sock.close()
    print("Connection closed.")


# print(radar_waveform.shape)  # 打印数组的形状
sample_rate = 4000000000  # 采样率（Hz）
time = np.arange(0, len(radar_waveform)) / sample_rate  # 生成时间轴

# 绘制波形
plt.figure(figsize=(10, 6))
plt.plot(time, radar_waveform)
plt.xlim([0, 1e-7])  # 设置 x 轴显示范围
plt.title('Received Radar Waveform')
plt.xlabel('Time (seconds)')
plt.ylabel('Amplitude')
plt.grid(True)
plt.show()

# 计算频谱
fft_data = np.fft.fft(radar_waveform)
frequencies = np.fft.fftfreq(len(radar_waveform), 1 / sample_rate)

# 只取正频率部分
positive_frequencies = frequencies[:len(frequencies) // 2]
magnitude = np.abs(fft_data[:len(frequencies) // 2])

# 绘制频谱
plt.figure(figsize=(10, 6))
plt.plot(positive_frequencies, magnitude)
plt.xlim([0, 10e9])  # 设置 x 轴显示范围
plt.title('Frequency Spectrum of Received Radar Waveform')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.grid(True)
plt.show()
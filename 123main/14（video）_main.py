import time
import os
import subprocess
import psutil
from ultralytics import YOLO
import numpy as np
import cv2
import csv

# 获取CPU和GPU占用情况
def get_system_usage():
    # CPU 占用
    cpu_percent = psutil.cpu_percent(interval=0.1)  # 更小的时间间隔，实时性更高
    cpu_core_usage = psutil.cpu_percent(percpu=True, interval=0.1)

    # CPU 时钟频率
    cpu_freq = psutil.cpu_freq()
    current_freq = cpu_freq.current if cpu_freq else None

    # GPU 显存占用
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,noheader,nounits'], stdout=subprocess.PIPE, text=True)
        gpu_memory = result.stdout.strip().split('\n')
        gpu_usage = []
        for mem in gpu_memory:
            used, total = map(int, mem.split(','))
            gpu_usage.append({'used': used, 'total': total, 'percent': round(used / total * 100, 2)})
    except Exception as e:
        gpu_usage = [{'error': str(e)}]  # 如果 `nvidia-smi` 不可用，返回错误

    return cpu_percent, cpu_core_usage, current_freq, gpu_usage

# 显示实时推理时间和系统资源
def display_usage(cpu_percent, cpu_core_usage, current_freq, gpu_usage):
    print(f"CPU Usage: {cpu_percent:.2f}%")
    print(f"Per-Core Usage: {', '.join([f'{core:.2f}%' for core in cpu_core_usage])}")
    if current_freq:
        print(f"CPU Current Frequency: {current_freq:.2f} MHz")
    else:
        print("CPU Frequency information not available.")
    if gpu_usage and 'error' not in gpu_usage[0]:
        for i, gpu in enumerate(gpu_usage):
            print(f"GPU {i} - Used: {gpu['used']} MB / {gpu['total']} MB ({gpu['percent']}%)")
    else:
        print("GPU usage information not available.")

# 加载YOLOv8模型
def load_yolov8_model(model_path):
    model = YOLO(model_path)  # 使用ultralytics的YOLO类加载YOLOv8模型
    return model

# 设置模型路径
pytorch_model_path = r"D:\123\123main\pt\best(AKConv).pt"  # 修改为你的YOLOv8模型路径

# 加载YOLOv8模型
model = load_yolov8_model(pytorch_model_path)

# 获取推理时间和系统占用情况
inference_time = []
cpu_percentages = []
gpu_usage = []
cpu_core_usages = []
cpu_frequencies = []

# 视频路径
video_path = r'C:\Users\niu\duckee.mkv'  # 修改为你的输入视频路径

# 打开视频文件
cap = cv2.VideoCapture(video_path)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 获取视频帧数

# 检查视频是否成功打开
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break  # 如果视频结束则跳出循环

    # 获取系统占用情况
    cpu_percent, cpu_core_usage, current_freq, gpu_usage_info = get_system_usage()

    # 实时显示系统占用
    display_usage(cpu_percent, cpu_core_usage, current_freq, gpu_usage_info)

    # 将占用情况记录下来
    cpu_percentages.append(cpu_percent)
    cpu_core_usages.append(cpu_core_usage)
    cpu_frequencies.append(current_freq)
    gpu_usage.append(gpu_usage_info)

    # 推理时间计时
    start_time = time.time()

    # 使用YOLOv8模型进行推理
    results = model(frame)  # YOLO 推理返回的是一个列表

    # 提取单个结果
    result = results[0]  # 取列表中的第一个元素，适用于单帧处理

    # 显示推理结果
    annotated_frame = np.array(result.plot())  # 将 YOLO 注释结果转换为 NumPy 数组
    cv2.imshow('YOLO Detection', annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # 按 'q' 键退出
        break

    end_time = time.time()
    inference_time.append(end_time - start_time)

# 释放视频捕捉对象
cap.release()
cv2.destroyAllWindows()

# 计算平均推理时间
average_inference_time = sum(inference_time) / len(inference_time)
print(f"\nInference time is {average_inference_time:.4f} seconds")
fps = 1.0 / average_inference_time
print(f"FPS is {fps:.2f}")

# 计算系统资源的平均值
average_cpu_percent = sum(cpu_percentages) / len(cpu_percentages)
average_cpu_core_usage = [sum(core) / len(core) for core in zip(*cpu_core_usages)]
average_cpu_frequency = sum(cpu_frequencies) / len(cpu_frequencies) if cpu_frequencies else None
average_gpu_usage = []

for gpu_index in range(len(gpu_usage[0])):  # 针对多 GPU
    avg_used = sum(frame_usage[gpu_index]['used'] for frame_usage in gpu_usage) / len(gpu_usage)
    avg_total = gpu_usage[0][gpu_index]['total']  # 假设每帧的总显存相同
    avg_percent = round(avg_used / avg_total * 100, 2)
    average_gpu_usage.append({'used': avg_used, 'total': avg_total, 'percent': avg_percent})

# 打印统计结果
print(f"\nAverage CPU Usage: {average_cpu_percent:.2f}%")
print(f"Average CPU Core Usage: {', '.join([f'{core:.2f}%' for core in average_cpu_core_usage])}")
if average_cpu_frequency:
    print(f"Average CPU Frequency: {average_cpu_frequency:.2f} MHz")
for i, gpu in enumerate(average_gpu_usage):
    print(f"GPU {i} - Avg Used: {gpu['used']:.2f} MB / {gpu['total']} MB ({gpu['percent']}%)")

# 将统计数据保存为 CSV 文件
csv_file_path = 'system_usage_summary.csv'
with open(csv_file_path, mode='w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    # 写入标题
    writer.writerow(['Metric', 'Value'])

    # 写入 CPU 数据
    writer.writerow(['Average CPU Usage (%)', f"{average_cpu_percent:.2f}"])
    for idx, core_usage in enumerate(average_cpu_core_usage):
        writer.writerow([f"Average CPU Core {idx + 1} Usage (%)", f"{core_usage:.2f}"])
    if average_cpu_frequency:
        writer.writerow(['Average CPU Frequency (MHz)', f"{average_cpu_frequency:.2f}"])

    # 写入 GPU 数据
    for i, gpu in enumerate(average_gpu_usage):
        writer.writerow([f"GPU {i} Avg Used Memory (MB)", f"{gpu['used']:.2f}"])
        writer.writerow([f"GPU {i} Total Memory (MB)", f"{gpu['total']}"])
        writer.writerow([f"GPU {i} Avg Usage (%)", f"{gpu['percent']:.2f}"])

print(f"\nSystem usage summary saved to {csv_file_path}")

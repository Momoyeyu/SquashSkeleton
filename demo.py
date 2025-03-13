import cv2
import os
from easy_ViTPose import VitInference

# 设置模型路径
model_path = './models/vitpose-b-aic.pth'
yolo_path = './models/yolov8l.pt'

# 初始化 VitInference 模型，启用视频跟踪
model = VitInference(model_path, yolo_path, model_name='b', yolo_size=640, is_video=False, device=None)

# 设置视频源
video_path = 'examples/video1.mp4'  # 如果使用摄像头，可改为 0
cap = cv2.VideoCapture(video_path)

# 检查视频是否成功打开
if not cap.isOpened():
    print(f"Error:fail to open video file {video_path}")
    exit()

# 获取视频文件名（不含路径）
video_filename = 'output_' + model_path.split('.')[-2].split('-')[-1] + '_' + os.path.basename(video_path)

# 构建输出路径
output_dir = './examples/outputs'
os.makedirs(output_dir, exist_ok=True)  # 确保输出目录存在
output_path = os.path.join(output_dir, video_filename)  # 输出路径为 ./examples/outputs/input.mp4
if os.path.exists(output_path):
    os.remove(output_path)  # make sure the previous output is deleted
print(f'output_path: {output_path}')

# 获取视频属性
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 创建 VideoWriter 对象
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用 mp4v 编解码器
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# 逐帧处理视频
while True:
    # 读取一帧
    ret, frame = cap.read()
    if not ret:
        print("Video is played out.")
        break

    # 将帧转换为 RGB 格式（模型需要 RGB 输入）
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 运行骨架检测
    keypoints = model.inference(frame_rgb)

    # 绘制骨架到帧上（返回 RGB 格式）
    frame_with_skeletons = model.draw(show_yolo=False, confidence_threshold=0.8)

    # 将结果转换回 BGR 格式以供 OpenCV 显示
    frame_with_skeletons_bgr = cv2.cvtColor(frame_with_skeletons, cv2.COLOR_RGB2BGR)

    out.write(frame_with_skeletons_bgr)

    # 显示处理后的帧
    cv2.imshow('Video', frame_with_skeletons_bgr)

    # 按 'q' 键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放视频捕获对象并关闭窗口
cap.release()
out.release()
cv2.destroyAllWindows()

# 重置模型跟踪器（在处理完视频后调用）
model.reset()

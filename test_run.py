import cv2
import numpy as np
import time
from feature_extraction import get_face_embedding


def test_my_module():
    print("正在启动摄像头，请看向镜头...")
    # 0 代表默认摄像头。如果你有外接摄像头或者虚拟摄像头，可能需要改成 1 或 2
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("错误：无法打开摄像头！请检查是否被其他程序（如微信、Zoom）占用。")
        return

    # 给摄像头 1 秒钟的预热时间，跳过开头的黑屏帧
    time.sleep(1)

    print("摄像头已就绪。")
    print("👉 按下键盘的 's' 键打印当前特征向量")
    print("👉 按下键盘的 'q' 键退出测试")

    while True:
        ret, frame = cap.read()

        if not ret:
            print("读取画面失败。")
            break

        try:
            # 调用你的核心提取函数
            embedding = get_face_embedding(frame)

            # 判断是否提取成功（不全为 0）
            if embedding.shape[0] > 0 and not np.all(embedding == 0):
                status_text = "Success! Face Detected"
                color = (0, 255, 0)  # 绿色
            else:
                status_text = "No Face Detected"
                color = (0, 0, 255)  # 红色

            # 在视频画面上打上状态文字
            cv2.putText(frame, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        except Exception as e:
            cv2.putText(frame, f"Error: {e}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # 弹出一个窗口显示实时画面
        cv2.imshow("Role 2 - Model Test", frame)

        # 捕捉键盘操作
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):  # 按 q 退出
            print("测试结束。")
            break
        elif key == ord('s'):  # 按 s 打印数据
            print("\n--- 手动触发提取 ---")
            if not np.all(embedding == 0):
                print(f"✅ 成功！特征向量维度: {embedding.shape}")
                print(f"向量前5位展示: {embedding[:5]}")
            else:
                print("⚠️ 当前画面中未检测到人脸，输出为全 0。")

    # 释放资源并关闭窗口
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    test_my_module()
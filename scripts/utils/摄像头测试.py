#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
📷 摄像头测试工具
用于验证摄像头是否正常工作
"""

import cv2

try:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise Exception("无法打开摄像头，请检查摄像头是否被占用")
    
    print("✅ 摄像头已打开，按 Q 键退出")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_canny = cv2.Canny(frame_gray, 50, 150)
        
        cv2.imshow("实时摄像头画面", frame)
        cv2.imshow("实时边缘检测", frame_canny)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("✅ 摄像头测试完成，已正常退出")

except Exception as e:
    print(f"❌ 摄像头测试失败：{e}")
    print("👉 排查建议：")
    print("   1. 检查摄像头是否被其他软件占用（如微信/钉钉）")
    print("   2. 管理员权限运行 VS Code")

import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np

# --- 頁面設定 ---
st.set_page_config(page_title="YOLOv8 物件偵測", layout="centered")
st.title("📹 YOLOv8 實時物件偵測")
st.write("上傳圖片以進行 YOLOv8 物件偵測")

# --- 載入模型 ---
@st.cache_resource
def load_model():
    # 第一次執行會自動下載 yolov8n.pt
    model = YOLO('yolov8n.pt') 
    return model

model = load_model()

# --- 側邊欄 ---
st.sidebar.header("設定")
confidence = st.sidebar.slider("偵測置信度 (Confidence)", 0.0, 1.0, 0.25)

# --- 檔案上傳 ---
uploaded_file = st.file_uploader("請選擇一張圖片...", type=["jpg", "jpeg", "png"])

# --- 製作確認按鈕 ---
run_detection = st.button("開始辨識")
if run_detection:
    if uploaded_file is None:
        st.warning("請先上傳圖片!!!")
    
    else:
        # 顯示上傳的圖片
        image = Image.open(uploaded_file)
        st.image(image, caption='上傳的圖片', use_container_width=True)
        
        with st.spinner("偵測中..."):
            # 轉換圖片格式進行預測
            img_array = np.array(image)
            
            # --- YOLOv8 預測 ---
            results = model(img_array, conf=confidence)
            
            # 繪製結果
            res_plotted = results[0].plot()
            
            # 顯示結果圖片
        st.image(res_plotted, caption='偵測結果', use_container_width=True)
    
    # 顯示偵測到的物件資訊
        with st.expander("查看偵測細節"):
            for box in results[0].boxes:
                st.write(f"類別: {model.names[int(box.cls[0])]}, 置信度: {box.conf[0]*100:.2f}%")


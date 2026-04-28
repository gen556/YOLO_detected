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

# --- 偵測邏輯 ---
if st.button("開始辨識"):
    if uploaded_file is None:
        st.warning("請先上傳圖片!!!")
    else:
        # 1. 使用 PIL 開啟圖片並強制轉為 RGB (處理 RGBA 問題)
        image = Image.open(uploaded_file).convert("RGB")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(image, caption='上傳的圖片', use_container_width=True)
        
        with st.spinner("偵測中..."):
            # 2. 轉換為 NumPy 陣列供 YOLO 使用
            img_array = np.array(image)
            
            # --- YOLOv8 預測 ---
            results = model(img_array, conf=confidence)
            
            # 3. 繪製結果 (注意：plot() 預設回傳 BGR 格式)
            res_plotted = results[0].plot()
            
    
            
        with col2:
            st.image(res_plotted, caption='偵測結果', use_container_width=True)
    
        # --- 顯示偵測到的物件資訊 ---
        with st.expander("查看偵測細節"):
            if len(results[0].boxes) > 0:
                for box in results[0].boxes:
                    label = model.names[int(box.cls[0])]
                    conf = box.conf[0] * 100
                    st.write(f"🔍 **{label}**: {conf:.2f}%")
            else:
                st.write("未偵測到任何物件。")
import streamlit as st
import cv2
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
import io
from PIL import Image

# 设置页面标题和布局
st.set_page_config(page_title="垃圾分类识别", layout="wide")
st.title("🗑️ 垃圾分类识别系统")
st.write("上传垃圾图片，AI会自动识别分类")

# 加载模型
@st.cache_resource
def load_classifier_model():
    try:
        model = load_model("realwaste_classifier.keras")
        st.success("模型加载成功!")
        return model
    except Exception as e:
        st.error(f"模型加载失败: {e}")
        return None

model = load_classifier_model()

# 设置参数
IMG_SIZE = (256, 256)
class_names = [
    'Cardboard', 'Food Organics', 'Glass', 'Metal',
    'Miscellaneous Trash', 'Paper', 'Plastic',
    'Textile Trash', 'Vegetation'
]

# 文件上传器
uploaded_file = st.file_uploader(
    "选择垃圾图片", 
    type=['jpg', 'jpeg', 'png'],
    help="上传一张垃圾图片进行分类识别"
)

if uploaded_file is not None and model is not None:
    # 读取图像
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    if img_bgr is not None:
        # 转换颜色空间
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, IMG_SIZE)
        
        # 预测
        input_batch = np.expand_dims(img_resized, 0)
        preds = model.predict(input_batch, verbose=0)[0]
        
        # 获取预测结果
        pred_idx = int(np.argmax(preds))
        pred_label = class_names[pred_idx]
        pred_conf = preds[pred_idx] * 100
        
        # 显示结果
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("识别结果")
            st.image(img_resized, use_column_width=True)
            st.success(f"**预测类别**: {pred_label}")
            st.info(f"**置信度**: {pred_conf:.1f}%")
        
        with col2:
            st.subheader("概率分布")
            
            # 创建条形图
            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.barh(class_names, preds, color='lightgray')
            bars[pred_idx].set_color('blue')
            ax.set_xlim(0, 1)
            ax.set_xlabel("概率")
            ax.set_title("各类别概率分布")
            
            # 在条形上添加概率值
            for i, (name, prob) in enumerate(zip(class_names, preds)):
                ax.text(prob + 0.01, i, f'{prob:.3f}', va='center')
            
            st.pyplot(fig)
        
        # 显示详细概率表格
        st.subheader("详细概率")
        for i, (name, prob) in enumerate(zip(class_names, preds)):
            percentage = prob * 100
            st.progress(float(prob), text=f"{name}: {percentage:.1f}%")
            
    else:
        st.error("无法读取图像文件")

elif uploaded_file is not None and model is None:
    st.error("模型未加载，无法进行预测")

# 添加使用说明
with st.expander("使用说明"):
    st.markdown("""
    ### 如何使用：
    1. 点击"选择垃圾图片"上传图像
    2. 支持格式: JPG, JPEG, PNG
    3. 系统会自动显示识别结果和概率分布
    
    ### 支持的垃圾类别：
    - 📦 Cardboard (纸板)
    - 🍎 Food Organics (厨余垃圾) 
    - 🥛 Glass (玻璃)
    - 🥫 Metal (金属)
    - 🗑️ Miscellaneous Trash (其他垃圾)
    - 📄 Paper (纸张)
    - 🧴 Plastic (塑料)
    - 👕 Textile Trash (纺织品垃圾)
    - 🌿 Vegetation (植物垃圾)
    """)

# 页脚
st.markdown("---")
st.caption("基于深度学习的垃圾分类识别系统 | 使用TensorFlow/Keras训练")
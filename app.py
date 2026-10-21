import streamlit as st
import cv2
import numpy as np
from PIL import Image
import plotly.graph_objects as go

# --- CẤU HÌNH TRANG WEB STREAMLIT ---
st.set_page_config(
    page_title="Hộp Oxyz Thông Minh - Smart City",
    page_icon="🏙️",
    layout="wide"
)

st.title("🏙️ HỘP OXYZ THÔNG MINH: ĐIỀU HÀNH KHÔNG PHẬN & GIAO HÀNG TỰ ĐỘNG SMART CITY")
st.markdown("""
Hệ thống **Bản sao số (Digital Twin)** tích hợp thuật toán **Thị giác máy tính OpenCV** và mô hình không gian 3D trực quan cho 4 nhiệm vụ Oxyz.
""")

st.markdown("---")

# --- HÀM XỬ LÝ OPENCV THÔNG MINH ---
def process_opencv_marker(image_np, scale_max=10.0):
    hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)
    
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])
    
    mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
    
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    h_img, w_img, _ = image_np.shape
    coord_val = 5.0 
    debug_img = image_np.copy()
    
    if contours:
        largest_c = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest_c) > 20:
            M = cv2.moments(largest_c)
            if M["m00"] > 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                
                if scale_max == 10.0:
                    coord_val = round((cX / w_img) * scale_max, 2)
                else:
                    coord_val = round(((h_img - cY) / h_img) * scale_max, 2)
                
                cv2.drawContours(debug_img, [largest_c], -1, (0, 255, 0), 3)
                cv2.circle(debug_img, (cX, cY), 6, (255, 0, 0), -1)
                
    return coord_val, debug_img

# --- PHẦN 1: TẢI 2 ẢNH VÀ XỬ LÝ OPENCV ---
st.header("📸 1. Trích Xuất Tọa Độ Tự Động Bằng OpenCV (Dual-Camera)")

col_img1, col_img2 = st.columns(2)

with col_img1:
    st.subheader("📷 Ảnh Top-down (Quét X, Y)")
    top_file = st.file_uploader("Tải ảnh mặt bằng sa bàn", type=["jpg", "jpeg", "png"], key="top_img")
    
with col_img2:
    st.subheader("📷 Ảnh Side-view (Quét cao độ Z)")
    side_file = st.file_uploader("Tải ảnh mặt đứng sa bàn", type=["jpg", "jpeg", "png"], key="side_img")

drone_x, drone_y, drone_z = 4.0, 4.0, 3.0

if top_file is not None and side_file is not None:
    img_top = Image.open(top_file)
    np_top = np.array(img_top)
    
    img_side = Image.open(side_file)
    np_side = np.array(img_side)
    
    drone_x, debug_top = process_opencv_marker(np_top, scale_max=10.0)
    h_t, w_t, _ = np_top.shape
    hsv_t = cv2.cvtColor(np_top, cv2.COLOR_RGB2HSV)
    mask_t = cv2.inRange(hsv_t, np.array([0, 120, 70]), np.array([10, 255, 255])) | cv2.inRange(hsv_t, np.array([170, 120, 70]), np.array([180, 255, 255]))
    contours_t, _ = cv2.findContours(mask_t, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours_t:
        lc_t = max(contours_t, key=cv2.contourArea)
        M_t = cv2.moments(lc_t)
        if M_t["m00"] > 0:
            cY_t = int(M_t["m01"] / M_t["m00"])
            drone_y = round(((h_t - cY_t) / h_t) * 10.0, 2)

    drone_z, debug_side = process_opencv_marker(np_side, scale_max=10.0)
    h_s, _, _ = np_side.shape
    contours_s, _ = cv2.findContours(cv2.inRange(cv2.cvtColor(np_side, cv2.COLOR_RGB2HSV), np.array([0, 120, 70]), np.array([10, 255, 255])), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours_s:
        lc_s = max(contours_s, key=cv2.contourArea)
        M_s = cv2.moments(lc_s)
        if M_s["m00"] > 0:
            cY_s = int(M_s["m01"] / M_s["m00"])
            drone_z = round(((h_s - cY_s) / h_s) * 10.0, 2)

    with col_img1:
        st.image(debug_top, caption="✅ OpenCV phát hiện tọa độ (X, Y)", use_column_width=True)
    with col_img2:
        st.image(debug_side, caption="✅ OpenCV phát hiện cao độ (Z)", use_column_width=True)

st.markdown("---")

# --- PHẦN 2: THIẾT LẬP TỌA ĐỘ VÀ KHỞI CHẠY ---
st.header("🎯 2. Tọa độ Không gian Đô thị Oxyz")

A = np.array([2.0, 3.0, 3.0])      # Trạm y tế A (Cố định)
B = np.array([8.0, 6.0, 5.0])      # Tòa nhà sự cố B (Cố định)
C = np.array([5.0, 1.0, 4.0])      # Chướng ngại vật C (Cố định)
K = np.array([drone_x, drone_y, drone_z]) # Vị trí Drone K (Động)

col_coord1, col_coord2, col_coord3, col_coord4 = st.columns(4)
col_coord1.metric("🏥 Trạm Y Tế A (Cố định)", "A(2; 3; 3)")
col_coord2.metric("🔥 Tòa Nhà Sự Cố B (Cố định)", "B(8; 6; 5)")
col_coord3.metric("🏢 Chướng Ngại Vật C (Cố định)", "C(5; 1; 4)")
col_coord4.metric("🛸 Drone K (OpenCV tự động)", f"K({drone_x}; {drone_y}; {drone_z})")

run_btn = st.button("🚀 KHỞI CHẠY HỆ THỐNG ĐIỀU HÀNH KHÔNG LƯU", type="primary", use_container_width=True)

if run_btn or top_file is not None:
    # --- PHẦN 3: TÍNH TOÁN 4 NHIỆM VỤ TOÁN HỌC ---
    vec_KA = A - K
    vec_AB = B - A
    dist_KA = np.linalg.norm(vec_KA)
    dist_AB = np.linalg.norm(vec_AB)
    total_dist = dist_KA + dist_AB
    est_time = total_dist * 2.5

    dist_direct_KB = np.linalg.norm(B - K)

    # Tính khoảng cách dùng tích vô hướng và hình chiếu vuông góc H trên đường thẳng AB
    vec_AC = C - A
    t_proj = np.dot(vec_AC, vec_AB) / np.dot(vec_AB, vec_AB)
    H_proj = A + t_proj * vec_AB # Tọa độ hình chiếu H
    vec_CH = H_proj - C
    dist_C_to_AB = np.linalg.norm(vec_CH)
    is_safe = dist_C_to_AB >= 1.5

    M_x = (A[0] - 2*K[0] + C[0]) / (1 - 2 + 1) if (1 - 2 + 1) != 0 else (A[0] + K[0] + C[0])/3
    M_y = (A[1] + K[1] + C[1])/3
    M = np.array([round(M_x, 2), round(M_y, 2), 0.0])
    
    vec_MA = A - M
    vec_MK = K - M
    vec_MC = C - M
    T_min = np.linalg.norm(vec_MA - 2*vec_MK + vec_MC)

    # --- PHẦN 4: VẼ BẢN ĐỒ KHÔNG GIAN 3D ---
    st.subheader("🌐 Bản Đồ Không Gian Đô Thị 3D (Digital Twin)")
    fig = go.Figure()

    points = {
        '🛸 Drone K': (K, '#00FFFF', 12, 'circle'),
        '🏥 Trạm Y Tế A': (A, '#00FF66', 10, 'diamond'),
        '🔥 Tòa Nhà Sự Cố B': (B, '#FF0055', 12, 'square'),
        '🏢 Chướng Ngại Vật C': (C, '#FF9900', 10, 'cross'),
        '📡 Trạm Radar M': (M, '#FFD700', 12, 'circle') # Màu vàng rực rỡ
    }
    
    for name, (coords, color, size, symbol) in points.items():
        fig.add_trace(go.Scatter3d(
            x=[coords[0]], y=[coords[1]], z=[coords[2]],
            mode='markers+text',
            marker=dict(size=size, color=color, symbol=symbol, line=dict(color='white', width=1)),
            text=[f"{name}<br>({coords[0]};{coords[1]};{coords[2]})"],
            textposition="top center", name=name
        ))

    fig.add_trace(go.Scatter3d(
        x=[K[0], A[0]], y=[K[1], A[1]], z=[K[2], A[2]],
        mode='lines+markers', line=dict(color='#00E5FF', width=6, dash='dash'),
        name='Chặng 1: K ➔ A (Lấy thuốc)'
    ))

    fig.add_trace(go.Scatter3d(
        x=[A[0], B[0]], y=[A[1], B[1]], z=[A[2], B[2]],
        mode='lines+markers', line=dict(color='#FF3D00', width=8),
        name='Chặng 2: A ➔ B (Giao hàng)'
    ))

    # 🟪 Đường màu Tím nối C với hình chiếu H trên đường thẳng AB
    fig.add_trace(go.Scatter3d(
        x=[C[0], H_proj[0]], y=[C[1], H_proj[1]], z=[C[2], H_proj[2]],
        mode='lines+markers', line=dict(color='#E040FB', width=6),
        name='Khoảng cách an toàn d(C, AB)'
    ))

    axis_len = 10
    fig.add_trace(go.Scatter3d(x=[0, axis_len], y=[0, 0], z=[0, 0], mode='lines', line=dict(color='red', width=6), name='Trục Ox'))
    fig.add_trace(go.Scatter3d(x=[0, 0], y=[0, axis_len], z=[0, 0], mode='lines', line=dict(color='lime', width=6), name='Trục Oy'))
    fig.add_trace(go.Scatter3d(x=[0, 0], y=[0, 0], z=[0, axis_len], mode='lines', line=dict(color='cyan', width=6), name='Trục Oz'))

    fig.update_layout(
        scene=dict(
            xaxis_title='Ox', yaxis_title='Oy', zaxis_title='Oz',
            xaxis=dict(range=[0, 10], backgroundcolor="#080812", gridcolor="#cccccc", gridwidth=2, zerolinecolor="#ff5555"),
            yaxis=dict(range=[0, 10], backgroundcolor="#080812", gridcolor="#cccccc", gridwidth=2, zerolinecolor="#55ff55"),
            zaxis=dict(range=[0, 10], backgroundcolor="#080812", gridcolor="#cccccc", gridwidth=2, zerolinecolor="#55ffff")
        ),
        margin=dict(r=0, l=0, b=0, t=0), height=650
    )

    st.plotly_chart(fig, use_container_width=True)

    # --- PHẦN 5: BÁO CÁO 4 NHIỆM VỤ (TINH GỌN & CHUẨN XÁC) ---
    st.markdown("---")
    st.header("📋 3. Báo Cáo Giải Toán 4 Nhiệm Vụ Oxyz")

    col_task1, col_task2 = st.columns(2)

    with col_task1:
        st.subheader("📍 Nhiệm vụ 1: Vectơ hành trình khẩn cấp")
        st.info(f"""
        • **Vectơ KA**: $\\vec{{KA}} = ({vec_KA[0]:.2f}; {vec_KA[1]:.2f}; {vec_KA[2]:.2f})$  
        • **Vectơ AB**: $\\vec{{AB}} = ({vec_AB[0]:.2f}; {vec_AB[1]:.2f}; {vec_AB[2]:.2f})$  
        • **Tổng quãng đường ($S$)**: $|\\vec{{KA}}| + |\\vec{{AB}}| = {total_dist:.2f}$ m  
        • **Thời gian đáp ứng**: $\\approx {est_time:.1f}$ giây  
        """)

        st.subheader("⚡ Nhiệm vụ 2: Tối ưu hóa năng lượng")
        st.success(f"""
        • **Hành lang bay kép ($K \\rightarrow A \\rightarrow B$)**: {total_dist:.2f} m  
        • **Đường bay thẳng trực tiếp ($K \\rightarrow B$)**: {dist_direct_KB:.2f} m  
        • *Kết luận*: Drone thực hiện tạt qua trạm $A$ lấy huyết thanh trước khi đến $B$.  
        """)

    with col_task2:
        st.subheader("🛡️ Nhiệm vụ 3: Hành lang an toàn (Tích vô hướng)")
        if is_safe:
            st.success(f"""
            • **Phương pháp**: Dùng tích vô hướng tìm hình chiếu $H$ của $C$ trên $AB$, với $t = \\frac{{\\vec{{AC}} \\cdot \\vec{{AB}}}}{{|\\vec{{AB}}|^2}}$  
            • **Khoảng cách thực tế**: $d(C, AB) = {dist_C_to_AB:.2f}$ m  
            • **Trạng thái**: ✅ **AN TOÀN TUYỆT ĐỐI** (Đường tím biểu diễn $CH$).  
            """)
        else:
            st.error(f"""
            • **Khoảng cách thực tế**: $d(C, AB) = {dist_C_to_AB:.2f}$ m  
            • **Trạng thái**: ⚠️ **CẢNH BÁO VA CHẠM!**  
            """)

        st.subheader("📡 Nhiệm vụ 4: Tối ưu trạm Radar mặt đất $M$")
        st.warning(f"""
        • **Biểu thức cực trị**: $T = |\\vec{{MA}} - 2\\vec{{MK}} + \\vec{{MC}}|$  
        • **Tọa độ trạm tối ưu (Màu Vàng)**: $M({M[0]}; {M[1]}; 0.0)$  
        • **Giá trị cực tiểu**: $T_{{\\min}} = {T_min:.2f}$  
        """)
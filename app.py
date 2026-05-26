import streamlit as st
import pandas as pd
import json
import os
from datetime import datetime

# Define workspace directory paths
WORKSPACE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(WORKSPACE_DIR, "data")

# Create data directory if it doesn't exist
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

st.set_page_config(
    page_title="Thống kê thực nghiệm DeFL_IDS",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium theme customization
st.markdown("""
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        * {
            font-family: 'Plus Jakarta Sans', sans-serif;
        }
        .main-header {
            background: linear-gradient(135deg, #00F2FE 0%, #4FACFE 50%, #7F00FF 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 800;
            font-size: 2.6rem;
            margin-bottom: 0.3rem;
            letter-spacing: -0.02em;
        }
        .sub-header {
            color: #8A99AD;
            font-size: 1.1rem;
            margin-bottom: 2rem;
            font-weight: 400;
        }
        .card {
            background-color: rgba(255, 255, 255, 0.02);
            border: 1px solid rgba(127, 0, 255, 0.15);
            border-radius: 16px;
            padding: 1.5rem;
            margin-bottom: 1rem;
            box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.04);
            backdrop-filter: blur(4px);
            -webkit-backdrop-filter: blur(4px);
            transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
        }
        .card:hover {
            transform: translateY(-3px);
            box-shadow: 0 15px 30px rgba(127, 0, 255, 0.1);
            border-color: rgba(0, 242, 254, 0.4);
        }
        .metric-title {
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 0.07em;
            color: #8892B0;
            margin-bottom: 0.4rem;
            font-weight: 600;
        }
        .metric-num {
            font-size: 2rem;
            font-weight: 700;
            background: linear-gradient(135deg, #00F2FE 0%, #4FACFE 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        /* Custom side header */
        .sidebar-header {
            font-weight: 700;
            font-size: 1.25rem;
            margin-bottom: 1.2rem;
            color: #1A202C;
            letter-spacing: -0.01em;
        }
        div[data-testid="stSidebar"] {
            background-color: #F8FAFC;
            border-right: 1px solid rgba(0, 0, 0, 0.05);
        }
        /* Dark mode fallback adjustment */
        @media (prefers-color-scheme: dark) {
            .sidebar-header {
                color: #F8FAFC;
            }
            div[data-testid="stSidebar"] {
                background-color: #0F172A;
                border-right: 1px solid rgba(255, 255, 255, 0.05);
            }
            .metric-num {
                background: linear-gradient(135deg, #00F2FE 0%, #A78BFA 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
            }
            .card {
                background-color: rgba(15, 23, 42, 0.6);
                border: 1px solid rgba(167, 139, 250, 0.15);
            }
        }
    </style>
""", unsafe_allow_html=True)

# Helper functions to manage files
def list_data_files():
    if not os.path.exists(DATA_DIR):
        return []
    files = [f for f in os.listdir(DATA_DIR) if f.endswith(".json") or f.endswith(".csv")]
    # sort by modification time (newest first)
    files.sort(key=lambda x: os.path.getmtime(os.path.join(DATA_DIR, x)), reverse=True)
    return files

def filter_allowed_columns(df):
    # Rename columns to their canonical names if they exist case-insensitively
    # We consolidate 'task' -> 'task_id', and 'round' -> 'round_id'
    
    # 1. Handle task / task_id case-insensitively
    task_col = None
    task_id_col = None
    for col in df.columns:
        if str(col).lower() == 'task':
            task_col = col
        elif str(col).lower() == 'task_id':
            task_id_col = col
            
    if task_col is not None:
        if task_id_col is None:
            df = df.rename(columns={task_col: 'task_id'})
        else:
            # both exist, drop task
            df = df.drop(columns=[task_col])

    # 2. Handle round / round_id case-insensitively
    round_col = None
    round_id_col = None
    for col in df.columns:
        if str(col).lower() == 'round':
            round_col = col
        elif str(col).lower() == 'round_id':
            round_id_col = col
            
    if round_col is not None:
        if round_id_col is None:
            df = df.rename(columns={round_col: 'round_id'})
        else:
            # both exist, drop round
            df = df.drop(columns=[round_col])

    # Now filter to allowed columns (canonical only)
    ALLOWED_COLS = [
        'task_id', 'round_id', 'train_loss', 'test_loss', 'accuracy', 
        'precision_macro', 'recall_macro', 'recall_marco', 
        'f1_macro', 'f1_marco', 'f1_weighted', 'f1_weight'
    ]
    cols_to_keep = []
    for allowed in ALLOWED_COLS:
        for col in df.columns:
            if str(col).lower() == allowed:
                if col not in cols_to_keep:
                    cols_to_keep.append(col)
                break
    if cols_to_keep:
        return df[cols_to_keep]
    return df

def save_data_file(file_name, data):
    file_path = os.path.join(DATA_DIR, file_name)
    if file_name.endswith(".csv"):
        if isinstance(data, pd.DataFrame):
            data.to_csv(file_path, index=False, encoding="utf-8")
        else:
            pd.DataFrame(data).to_csv(file_path, index=False, encoding="utf-8")
    else:
        # default JSON
        with open(file_path, "w", encoding="utf-8") as f:
            if isinstance(data, pd.DataFrame):
                records = data.to_dict(orient="records")
                json.dump(records, f, indent=2, ensure_ascii=False)
            else:
                json.dump(data, f, indent=2, ensure_ascii=False)

def load_data_file(file_name):
    file_path = os.path.join(DATA_DIR, file_name)
    if not os.path.exists(file_path):
        return pd.DataFrame()
    
    if file_name.endswith(".csv"):
        df = pd.read_csv(file_path)
    else:
        # JSON
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            df = pd.DataFrame(data)
        elif isinstance(data, dict):
            try:
                df = pd.DataFrame(data)
            except Exception:
                try:
                    df = pd.json_normalize(data)
                except Exception:
                    df = pd.DataFrame([data])
        else:
            df = pd.DataFrame([{"Value": data}])
            
    return filter_allowed_columns(df)

def delete_data_file(file_name):
    file_path = os.path.join(DATA_DIR, file_name)
    if os.path.exists(file_path):
        os.remove(file_path)
        return True
    return False

# Main Layout
# Sidebar
with st.sidebar:
    st.markdown('<div class="sidebar-header">📥 Tải Lên & Quản Lý</div>', unsafe_allow_html=True)
    
    # 1. Upload File (JSON or CSV)
    uploaded_file = st.file_uploader(
        "Chọn file JSON hoặc CSV", 
        type=["json", "csv"], 
        help="Kéo thả hoặc nhấn để chọn file JSON hoặc CSV"
    )
    
    if uploaded_file is not None:
        try:
            file_name = uploaded_file.name
            # Simple sanitize
            file_name = "".join(c for c in file_name if c.isalnum() or c in ('.', '_', '-'))
            
            if file_name.endswith(".json"):
                file_data = json.load(uploaded_file)
                save_data_file(file_name, file_data)
            elif file_name.endswith(".csv"):
                file_df = pd.read_csv(uploaded_file)
                save_data_file(file_name, file_df)
                
            st.success(f"Đã lưu file: {file_name}")
            # rerun to update the file list
            st.rerun()
        except json.JSONDecodeError as e:
            st.error(f"Lỗi cú pháp JSON: {str(e)}")
        except Exception as e:
            st.error(f"Lỗi khi lưu file: {str(e)}")

    st.markdown("---")
    
    # 2. File Selection & Deletion
    data_files = list_data_files()
    
    if data_files:
        st.markdown('<div class="sidebar-header">📂 Danh Sách File</div>', unsafe_allow_html=True)
        
        # Display list of files to select
        selected_file = st.selectbox(
            "Chọn file để hiển thị",
            options=data_files,
            index=0,
            key="selected_file_dropdown"
        )
        
        # Deletion interface
        st.markdown(" ")
        confirm_delete = st.checkbox("Xác nhận muốn xóa", key="confirm_delete_checkbox", help="Tick vào đây để hiển thị nút xóa")
        
        if confirm_delete:
            if st.button("🗑️ Xóa File Này", type="primary", use_container_width=True):
                if delete_data_file(selected_file):
                    st.success(f"Đã xóa file: {selected_file}")
                    # reset confirmation and rerun
                    st.rerun()
                else:
                    st.error("Không thể xóa file.")
    else:
        st.info("Chưa có file nào. Hãy tải lên file đầu tiên ở trên!")
        selected_file = None

# Main Panel
st.markdown('<div class="main-header">🛡️ THỐNG KÊ THỰC NGHIỆM DeFL_IDS</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Hệ thống phân tích, hiển thị và so sánh kết quả thực nghiệm học máy liên kết phát hiện xâm nhập (DeFL_IDS).</div>', unsafe_allow_html=True)

# Setup Tabs
tab_detail, tab_summary = st.tabs(["📄 Chi Tiết Từng File", "📈 Đối Sánh Hiệu Năng (Round 19)"])

with tab_detail:
    if not selected_file:
        # Beautiful welcome/empty state
        st.markdown("""
            <div style="text-align: center; padding: 4rem 2rem; border: 2px dashed rgba(127, 0, 255, 0.2); border-radius: 20px; background-color: rgba(255, 255, 255, 0.01);">
                <span style="font-size: 5rem;">🛡️</span>
                <h3 style="margin-top: 1.5rem; font-weight: 600; background: linear-gradient(135deg, #00F2FE 0%, #7F00FF 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">Hệ Thống Thống Kê Thực Nghiệm DeFL_IDS</h3>
                <p style="color: #8A99AD; max-width: 600px; margin: 0.5rem auto 1.5rem auto;">
                    Chào mừng bạn đến với bảng điều khiển phân tích DeFL_IDS. Vui lòng tải lên tệp kết quả thực nghiệm (định dạng JSON hoặc CSV) ở thanh bên để bắt đầu phân tích chi tiết và đối sánh dữ liệu.
                </p>
            </div>
        """, unsafe_allow_html=True)
    else:
        # Load selected file data
        file_path = os.path.join(DATA_DIR, selected_file)
        try:
            df = load_data_file(selected_file)
            
            # Display File Summary
            st.markdown(f"### 📄 Đang xem: `{selected_file}`")
            
            # Styled Metrics
            col1, col2, col3 = st.columns(3)
            file_size_kb = os.path.getsize(file_path) / 1024
            
            with col1:
                st.markdown(f"""
                    <div class="card">
                        <div class="metric-title">Số dòng dữ liệu</div>
                        <div class="metric-num">{len(df)}</div>
                    </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                    <div class="card">
                        <div class="metric-title">Số cột dữ liệu</div>
                        <div class="metric-num">{len(df.columns)}</div>
                    </div>
                """, unsafe_allow_html=True)
            with col3:
                st.markdown(f"""
                    <div class="card">
                        <div class="metric-title">Kích thước file</div>
                        <div class="metric-num">{file_size_kb:.2f} KB</div>
                    </div>
                """, unsafe_allow_html=True)

            st.markdown("---")
            
            # 3. Interactive Data Editor
            st.markdown("#### 📝 Bảng Dữ Liệu Tương Tác")
            st.caption("Mẹo: Nhấp đúp vào ô để sửa dữ liệu, nhấp vào tiêu đề cột để sắp xếp, hoặc kéo thả các cột.")
            
            # Use data editor to let user edit values
            edited_df = st.data_editor(
                df,
                use_container_width=True,
                num_rows="dynamic",  # allow adding/deleting rows
                key="data_editor_instance"
            )
            
            # Export Actions and Save changes
            col_s1, col_s2, col_s3 = st.columns([1, 1, 2])
            with col_s1:
                if st.button("💾 Lưu các chỉnh sửa", type="primary", use_container_width=True, help="Lưu đè các sửa đổi vào file đang mở"):
                    save_data_file(selected_file, edited_df)
                    st.success("Đã lưu các chỉnh sửa thành công!")
                    st.rerun()
                    
            with col_s2:
                # Download options
                csv_data = edited_df.to_csv(index=False).encode('utf-8')
                if selected_file.endswith(".json"):
                    csv_download_name = selected_file.replace(".json", ".csv")
                else:
                    csv_download_name = selected_file
                st.download_button(
                    label="📥 Tải xuống dạng CSV",
                    data=csv_data,
                    file_name=csv_download_name,
                    mime="text/csv",
                    use_container_width=True
                )
                
            with col_s3:
                edited_list = edited_df.to_dict(orient="records")
                json_str = json.dumps(edited_list, indent=2, ensure_ascii=False).encode('utf-8')
                if selected_file.endswith(".csv"):
                    json_download_name = selected_file.replace(".csv", ".json")
                else:
                    json_download_name = selected_file
                st.download_button(
                    label="📥 Tải xuống dạng JSON",
                    data=json_str,
                    file_name=json_download_name,
                    mime="application/json",
                    use_container_width=True
                )

            st.markdown("---")

            # 4. Data Visualization / Charts
            numeric_cols = edited_df.select_dtypes(include=['number']).columns.tolist()
            
            if len(numeric_cols) > 0:
                st.markdown("#### 📈 Phân Tích & Vẽ Biểu Đồ Tự Động")
                
                # Select axes
                col_v1, col_v2 = st.columns(2)
                all_cols = edited_df.columns.tolist()
                with col_v1:
                    x_axis = st.selectbox("Chọn trục X (Ngang)", options=all_cols, index=0)
                
                with col_v2:
                    default_y_idx = 0
                    if len(numeric_cols) > 1 and x_axis in numeric_cols:
                        non_x_numeric = [c for c in numeric_cols if c != x_axis]
                        if non_x_numeric:
                            default_y_idx = numeric_cols.index(non_x_numeric[0])
                    
                    y_axis = st.selectbox("Chọn cột vẽ biểu đồ (Trục Y - Số)", options=numeric_cols, index=default_y_idx)
                
                chart_type = st.radio("Loại biểu đồ", options=["Biểu đồ đường (Line)", "Biểu đồ cột (Bar)", "Biểu đồ vùng (Area)"], horizontal=True)
                
                # Plot the chart
                chart_data = edited_df[[x_axis, y_axis]].dropna()
                
                if not chart_data.empty:
                    chart_data = chart_data.sort_values(by=x_axis)
                    if "Line" in chart_type:
                        st.line_chart(chart_data.set_index(x_axis))
                    elif "Bar" in chart_type:
                        st.bar_chart(chart_data.set_index(x_axis))
                    else:
                        st.area_chart(chart_data.set_index(x_axis))
                else:
                    st.warning("Không có dữ liệu hợp lệ để vẽ biểu đồ.")
            else:
                st.info("Không phát hiện cột dữ liệu số nào để hiển thị biểu đồ phân tích.")
                
        except Exception as e:
            st.error(f"Lỗi khi hiển thị file: {str(e)}")

with tab_summary:
    st.markdown("### 📈 Đối Sánh Hiệu Năng Thực Nghiệm (Round 19)")
    st.write("Bảng so sánh kết quả huấn luyện mô hình DeFL_IDS từ tất cả các tệp thực nghiệm đã lưu tại local round thứ 19.")
    
    all_files = list_data_files()
    if not all_files:
        st.info("Chưa có file nào được tải lên để tổng hợp thống kê.")
    else:
        summary_rows = []
        for file in all_files:
            try:
                temp_df = load_data_file(file)
                if temp_df.empty:
                    continue
                
                # Check if 'round_id' and 'task_id' columns exist
                round_col = 'round_id' if 'round_id' in temp_df.columns else None
                task_col = 'task_id' if 'task_id' in temp_df.columns else None
                
                if round_col is not None:
                    temp_df[round_col] = pd.to_numeric(temp_df[round_col], errors='coerce')
                    
                    if task_col is not None:
                        # Filter for local round 19 of each task (i.e. round where round - min_round == 19)
                        filtered_groups = []
                        for task_val, group in temp_df.groupby(task_col):
                            group_sorted = group.sort_values(by=round_col)
                            if not group_sorted.empty:
                                min_r = group_sorted[round_col].min()
                                row_match = group_sorted[group_sorted[round_col] - min_r == 19]
                                if not row_match.empty:
                                    filtered_groups.append(row_match)
                                elif len(group_sorted) > 19:
                                    filtered_groups.append(group_sorted.iloc[[19]])
                        if filtered_groups:
                            filtered_df = pd.concat(filtered_groups)
                        else:
                            filtered_df = pd.DataFrame()
                    else:
                        # No task column, filter where round - min_round == 19, or literal 19
                        min_r = temp_df[round_col].min()
                        row_match = temp_df[temp_df[round_col] - min_r == 19]
                        if not row_match.empty:
                            filtered_df = row_match
                        else:
                            filtered_df = temp_df[temp_df[round_col] == 19].copy()
                    
                    if not filtered_df.empty:
                        # Add Source File column
                        filtered_df.insert(0, "File Nguồn", file)
                        summary_rows.append(filtered_df)
            except Exception as e:
                st.warning(f"Không thể xử lý file `{file}`: {str(e)}")
        
        if summary_rows:
            combined_summary_df = pd.concat(summary_rows, ignore_index=True)
            
            # Remove 'round_id' column from summary table
            if 'round_id' in combined_summary_df.columns:
                combined_summary_df = combined_summary_df.drop(columns=['round_id'])
            
            # Display stats cards
            col_stat1, col_stat2 = st.columns(2)
            with col_stat1:
                st.markdown(f"""
                    <div class="card">
                        <div class="metric-title">Tổng số file có dữ liệu Round 19</div>
                        <div class="metric-num">{combined_summary_df['File Nguồn'].nunique()} / {len(all_files)}</div>
                    </div>
                """, unsafe_allow_html=True)
            with col_stat2:
                st.markdown(f"""
                    <div class="card">
                        <div class="metric-title">Tổng số dòng ghi nhận tại Round 19</div>
                        <div class="metric-num">{len(combined_summary_df)}</div>
                    </div>
                """, unsafe_allow_html=True)
                
            st.markdown("#### Bảng dữ liệu tổng hợp (Round 19)")
            st.dataframe(combined_summary_df, use_container_width=True)
            
            # Export buttons
            col_exp1, col_exp2 = st.columns(2)
            with col_exp1:
                csv_summary = combined_summary_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Tải xuống Báo cáo tổng hợp (CSV)",
                    data=csv_summary,
                    file_name="summary_round_19.csv",
                    mime="text/csv",
                    key="download_summary_csv"
                )
            with col_exp2:
                json_summary = combined_summary_df.to_json(orient="records", force_ascii=False, indent=2).encode('utf-8')
                st.download_button(
                    label="📥 Tải xuống Báo cáo tổng hợp (JSON)",
                    data=json_summary,
                    file_name="summary_round_19.json",
                    mime="application/json",
                    key="download_summary_json"
                )
                
            # Visualization for round 19 across files
            numeric_cols = combined_summary_df.select_dtypes(include=['number']).columns.tolist()
            # Remove keys like round_id/task_id from choices
            for col in list(numeric_cols):
                if str(col).lower() in ['round_id', 'task_id']:
                    numeric_cols.remove(col)
                    
            if len(numeric_cols) > 0:
                st.markdown("---")
                st.markdown("#### 📊 So Sánh Các File Tại Round 19")
                y_axis_summary = st.selectbox(
                    "Chọn cột chỉ số để so sánh giữa các file",
                    options=numeric_cols,
                    key="y_axis_summary_selectbox"
                )
                
                chart_df = combined_summary_df.copy()
                
                # Check for task_id column
                task_col = 'task_id' if 'task_id' in chart_df.columns else None
                
                if task_col is not None:
                    chart_df['Label'] = chart_df['File Nguồn'] + " (Task " + chart_df[task_col].astype(str) + ")"
                else:
                    chart_df['Label'] = chart_df['File Nguồn']
                
                comparison_data = chart_df[['Label', y_axis_summary]].dropna()
                if not comparison_data.empty:
                    st.bar_chart(comparison_data.set_index('Label'))
                else:
                    st.warning("Không có dữ liệu hợp lệ để so sánh.")
        else:
            st.warning("Không tìm thấy dữ liệu có `round` là 19 trong bất kỳ file nào hiện tại.")

import streamlit as st

# Tiêu đề
st.title("Ứng dụng quản lý công việc")

# Khởi tạo session state để lưu danh sách công việc
if 'tasks' not in st.session_state:
    st.session_state.tasks = []

# Nhập thông tin công việc
task_name = st.text_input("Tên công việc:")
priority = st.slider("Mức độ ưu tiên:", 1, 5, 1)
status = st.selectbox("Trạng thái:", ["Chưa làm", "Đang làm", "Hoàn thành"])

# Nút thêm công việc
if st.button("Thêm công việc"):
    if task_name:
        st.session_state.tasks.append({"Tên": task_name, "Ưu tiên": priority, "Trạng thái": status})
        st.success("Đã thêm công việc!")
    else:
        st.warning("Vui lòng nhập tên công việc!")

# Hiển thị danh sách công việc
if st.session_state.tasks:
    st.subheader("Danh sách công việc")
    for idx, task in enumerate(st.session_state.tasks):
        st.text(f"{idx + 1}. {task['Tên']} - Ưu tiên: {task['Ưu tiên']} - Trạng thái: {task['Trạng thái']}")

    # Nút xóa danh sách
    if st.button("Xóa danh sách"):
        st.session_state.tasks = []
        st.warning("Đã xóa toàn bộ công việc!")

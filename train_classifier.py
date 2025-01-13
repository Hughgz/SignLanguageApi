import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Load dữ liệu từ file
try:
    data_dict = pickle.load(open('./data.pickle', 'rb'))
except FileNotFoundError:
    raise FileNotFoundError("File 'data.pickle' không tồn tại. Vui lòng kiểm tra đường dẫn.")

if 'data' not in data_dict or 'labels' not in data_dict:
    raise KeyError("'data' hoặc 'labels' không có trong file pickle.")

data = data_dict['data']
labels = data_dict['labels']

# Xử lý dữ liệu: lọc và chuẩn hóa
processed_data = []
for i, item in enumerate(data):
    try:
        if hasattr(item, '__len__') and len(item) == 42:
            processed_data.append([float(x) for x in item])
        else:
            print(f"Invalid item at index {i}: {item}")
    except ValueError:
        print(f"Non-numeric value at index {i}: {item}")

data = np.asarray(processed_data)
labels = np.asarray(labels[:len(processed_data)])  # Đảm bảo nhãn khớp với dữ liệu

if len(data) != len(labels):
    raise ValueError(f"Số lượng dữ liệu ({len(data)}) không khớp với số lượng nhãn ({len(labels)}).")

# Chia dữ liệu thành tập train và test
x_train, x_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, shuffle=True, stratify=labels
)

print(f"Số lượng mẫu huấn luyện: {len(x_train)}")
print(f"Số lượng mẫu kiểm tra: {len(x_test)}")

# Huấn luyện mô hình
model = RandomForestClassifier(random_state=42)
model.fit(x_train, y_train)

# Dự đoán
y_predict = model.predict(x_test)

# Đánh giá
score = accuracy_score(y_test, y_predict)
print(f"Độ chính xác: {score * 100:.2f}%")
print("\nBáo cáo phân loại:")
print(classification_report(y_test, y_predict))

# Lưu mô hình
with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)

print("Mô hình đã được lưu thành công vào 'model.p'.")

import os
import cv2

DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

number_of_classes = 31  # Số lớp cần thu thập
dataset_size = 100     # Số lượng ảnh cần thu thập cho mỗi lớp

cap = cv2.VideoCapture(0)

for j in range(number_of_classes):
    class_dir = os.path.join(DATA_DIR, str(j))
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    # Kiểm tra số lượng ảnh hiện có trong thư mục lớp
    existing_images = len(os.listdir(class_dir))
    print(f'Class {j} already has {existing_images} images.')

    # Bắt đầu từ ảnh tiếp theo
    counter = existing_images

    if counter >= dataset_size:
        print(f'Class {j} has already reached the target dataset size.')
        continue

    print(f'Collecting data for class {j} starting from image {counter}.')

    # Hiển thị thông báo sẵn sàng
    while True:
        ret, frame = cap.read()
        cv2.putText(frame, 'Ready? Press "Q" to start collecting!', (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) == ord('q'):
            break

    # Thu thập dữ liệu
    while counter < dataset_size:
        ret, frame = cap.read()
        if not ret:
            print("Error reading from camera. Exiting...")
            break

        cv2.imshow('frame', frame)
        cv2.waitKey(25)

        # Lưu ảnh với số thứ tự tiếp theo
        cv2.imwrite(os.path.join(class_dir, f'{counter}.jpg'), frame)
        counter += 1

    print(f'Finished collecting data for class {j}.')

cap.release()
cv2.destroyAllWindows()

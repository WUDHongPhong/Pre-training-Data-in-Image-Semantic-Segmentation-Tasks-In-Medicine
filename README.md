Tải data từ web Kaggle: https://www.kaggle.com/datasets/pkdarabi/brain-tumor-image-dataset-semantic-segmentation?resource=download.
Sau đó tạo file Data và thay đổi đường dẫn trong code, sau khi chạy code sẽ tạo được thêm 2 folder (masks/(train,test,valid) và processed/(train,test,valid)).

Pipline: 
- Xây dựng Dataset class: Tạo một class tùy chỉnh trong PyTorch để tải, tiền xử lý và trả về từng cặp ảnh/mask.
- Tiền xử lý chi tiết: Trong Dataset class, chúng ta sẽ thực hiện:
    + Đọc ảnh và mask.
    + Resize (Thay đổi kích thước) ảnh và mask về cùng một kích thước chuẩn. -> Dataset đã resize sẵn
    + Áp dụng Data Augmentation (thực hiện các phép biến đổi ngẫu nhiên trên NumPy array đã resize).
    + Chuẩn hóa (Normalize) giá trị pixel của ảnh.  
    + Chuyển đổi ảnh và mask sang dạng Tensor của PyTorch.
- Tạo DataLoader: Sử dụng DataLoader để tạo các batch dữ liệu, giúp cho việc training hiệu quả hơn.

- Kiểm tra và trực quan hóa: Lấy một batch dữ liệu từ DataLoader và hiển thị để đảm bảo mọi thứ đã được xử lý chính xác.


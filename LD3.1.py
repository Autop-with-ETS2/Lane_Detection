import cv2
import numpy as np

# Sobel 필터 적용 함수
def sobel_xy(img, orient='x', thresh=(20, 100)):
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(img, cv2.CV_64F, 1, 0))
    else:
        abs_sobel = np.absolute(cv2.Sobel(img, cv2.CV_64F, 0, 1))
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel)) if np.max(abs_sobel) != 0 else np.zeros_like(img)
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 255
    return binary_output

# 프레임을 전처리하고 흰색에 가까운 색만 남기기
def preprocess_frame(frame):
    # 그레이스케일 변환
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 가우시안 블러 적용
    blurred_frame = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 소벨 필터 적용
    sobel_x = sobel_xy(blurred_frame, 'x', (35, 100))
    sobel_y = sobel_xy(blurred_frame, 'y', (30, 255))

    # 그라디언트 결합
    grad_combine = np.zeros_like(sobel_x).astype(np.uint8)
    grad_combine[((sobel_x > 1) & (sobel_y > 1))] = 255

    # Canny 엣지 검출
    edges = cv2.Canny(blurred_frame, 50, 150)

    # Sobel과 Canny 결합
    combined = cv2.bitwise_or(grad_combine, edges)

    return combined

# 세로선을 검출하는 함수
def detect_vertical_lines(frame):
    processed_frame = preprocess_frame(frame)
    
    height, width = processed_frame.shape
    
    # ROI 영역 설정 (사다리꼴 형태)
    mask = np.zeros_like(processed_frame)
    roi_vertices = np.array([[ 
        (380, 510),  # 왼쪽 아래
        (380, 300),  # 왼쪽 위
        (660, 300),  # 오른쪽 위
        (720, 510)   # 오른쪽 아래
    ]], dtype=np.int32)
    
    # ROI 영역 시각화 (원본 프레임에 사다리꼴을 빨간색으로 표시)
    cv2.polylines(frame, [roi_vertices], isClosed=True, color=(0, 0, 255), thickness=3)

    # ROI 마스크를 적용하여 ROI 영역 내의 엣지만 남기기
    cv2.fillPoly(mask, [roi_vertices], 255)
    masked_edges = cv2.bitwise_and(processed_frame, mask)

    # 흰색(255) 픽셀의 좌표를 가져옴
    white_pixels = np.argwhere(masked_edges == 255)

    # 원본 이미지에 흰색 픽셀을 초록색으로 오버레이, 굵기 조절을 위해 원 대신 작은 원을 그림
    for pixel in white_pixels:
        y, x = pixel
        cv2.circle(frame, (x, y), 2, (0, 255, 0), thickness=-1)  # 원을 그려 굵기 2로 확대
    


    # 검출된 엣지를 확인하기 위한 디버그용 시각화
    cv2.imshow('Masked Edges', masked_edges)

    return frame

# 영상을 재생하고 선 검출 및 ROI 영역을 표시하는 함수
def show_video_processing(video_path):
    cv2.namedWindow('Processed Video', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Processed Video', 800, 600)

    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # 프레임에서 차선 인식 및 ROI 시각화
        lane_frame = detect_vertical_lines(frame)
        
        # 화면에 표시
        cv2.imshow('Processed Video', lane_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = "C:\\Users\\girookim\\Desktop\\Autopiliot\\test2.mp4"
    video_path2 = "C:\\Users\\girookim\\Desktop\\Autopiliot\\test.mp4"
    show_video_processing(video_path)
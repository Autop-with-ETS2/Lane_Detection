import cv2
import numpy as np
import math

# 차선 검출 함수
def detect_lane_lines(frame):
    # 1. 그레이스케일 변환
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 2. 가우시안 블러 적용 (노이즈 제거)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 3. Canny Edge Detection (엣지 검출)
    edges = cv2.Canny(blur, 30, 120)
    
    # 4. 중앙에 위치한 ROI 설정
    vertices = np.array([[(312, 184), (812, 184), (812, 568), (312, 568)]], dtype=np.int32)
    mask = np.zeros_like(edges)
    cv2.fillPoly(mask, vertices, 255)
    masked_edges = cv2.bitwise_and(edges, mask)

    # 마스킹된 결과 출력 (크기 축소)
    resized_masked_edges = cv2.resize(masked_edges, (frame.shape[1] // 2, frame.shape[0] // 2))
    cv2.imshow('Masked Edges', resized_masked_edges)

    # 5. 허프 변환을 이용한 직선 검출
    lines = cv2.HoughLinesP(masked_edges, 1, np.pi / 180, threshold=50, minLineLength=5, maxLineGap=100)

    # 6. 차선 좌표 계산 및 시각화 (선으로 표시)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # 기울기와 각도 계산
            if x2 - x1 != 0:
                slope = (y2 - y1) / (x2 - x1)
                angle = math.degrees(math.atan(slope))
                # 거의 가로인 선 배제 (기울기가 -5~5도 사이인 선)
                if -5 <= angle <= 5:
                    continue
            # 선 그리기
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # ROI 시각화 - 중앙 사각형을 파란색으로 표시
    cv2.polylines(frame, vertices, isClosed=True, color=(135, 206, 250), thickness=2)
    
    # 결과 반환
    return frame, edges

# 비디오 처리 함수
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 차선 검출
        lane_frame, edge = detect_lane_lines(frame)

        # 결과 출력
        cv2.imshow('Lane Detection', lane_frame)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# MP4 파일 경로
video_path = "C:\\Users\\girookim\\Desktop\\test3.mp4"

process_video(video_path)

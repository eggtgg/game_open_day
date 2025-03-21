import mediapipe as mp
import cv2
import math

class myPose:
    # Định nghĩa các hằng số ngưỡng
    CLAP_THRESHOLD = 50
    JUMP_THRESHOLD = 10
    DOWN_THRESHOLD = 50

    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose()
        self.mp_drawing = mp.solutions.drawing_utils
        self.shoulder_line_y = 0  # Lưu vị trí đường ngang của vai khi vỗ tay bắt đầu game

    def detectPose(self, image):
        # Chuyển ảnh sang RGB để xử lý
        imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(imageRGB)

        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                image, landmark_list=results.pose_landmarks,
                connections=self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing.DrawingSpec(color=(255, 225, 255),
                                                                   thickness=3,
                                                                   circle_radius=3),
                connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(0, 0, 255),
                                                                     thickness=2)
            )
        return image, results

    def _get_image_coords(self, landmark, image_shape):
        image_height, image_width, _ = image_shape
        return int(landmark.x * image_width), int(landmark.y * image_height)

    def checkPose_LRC(self, image, results):
        image_height, image_width, _ = image.shape
        mid_width = image_width // 2

        leftShoulder = results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        rightShoulder = results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        leftShoulder_x, _ = self._get_image_coords(leftShoulder, image.shape)
        rightShoulder_x, _ = self._get_image_coords(rightShoulder, image.shape)

        if leftShoulder_x < mid_width and rightShoulder_x < mid_width:
            LRC = "L"
        elif leftShoulder_x > mid_width and rightShoulder_x > mid_width:
            LRC = "R"
        else:
            LRC = "C"

        cv2.putText(image, LRC, (5, image_height - 10), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 3)
        cv2.line(image, (mid_width, 0), (mid_width, image_height), (255, 255, 255), 2)
        return image, LRC

    def checkPose_JSD(self, image, results):
        image_height, image_width, _ = image.shape

        leftShoulder = results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        rightShoulder = results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        _, leftShoulder_y = self._get_image_coords(leftShoulder, image.shape)
        _, rightShoulder_y = self._get_image_coords(rightShoulder, image.shape)
        centerShoulder_y = (leftShoulder_y + rightShoulder_y) // 2

        if centerShoulder_y < self.shoulder_line_y - self.JUMP_THRESHOLD:
            JSD = "J"
        elif centerShoulder_y > self.shoulder_line_y + self.DOWN_THRESHOLD:
            JSD = "D"
        else:
            JSD = "S"

        cv2.putText(image, JSD, (5, image_height - 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 255), 3)
        cv2.line(image, (0, self.shoulder_line_y), (image_width, self.shoulder_line_y), (0, 255, 255), 2)
        return image, JSD

    def checkPose_Clap(self, image, results):
        image_height, image_width, _ = image.shape

        left_hand = results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_WRIST]
        right_hand = results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_WRIST]
        left_coords = self._get_image_coords(left_hand, image.shape)
        right_coords = self._get_image_coords(right_hand, image.shape)

        distance = int(math.hypot(left_coords[0] - right_coords[0], left_coords[1] - right_coords[1]))

        CLAP = "C" if distance < self.CLAP_THRESHOLD else "N"
        cv2.putText(image, CLAP, (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 0), 3)
        return image, CLAP

    def save_shoulder_line_y(self, image, results):
        image_height, _, _ = image.shape

        leftShoulder = results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        rightShoulder = results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        _, left_y = self._get_image_coords(leftShoulder, image.shape)
        _, right_y = self._get_image_coords(rightShoulder, image.shape)
        self.shoulder_line_y = (left_y + right_y) // 2

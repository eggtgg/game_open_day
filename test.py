import cv2
import pyautogui
from myPose import myPose

class MyGame:
    # Định nghĩa các hằng số vị trí
    LEFT, CENTER, RIGHT = 0, 1, 2
    DOWN, STAND, JUMP = 0, 1, 2
    CLAP_THRESHOLD = 10  # Số frame cần để kích hoạt hành động từ vỗ tay

    def __init__(self):
        self.pose = myPose()
        self.game_started = False
        self.x_position = self.CENTER
        self.y_position = self.STAND
        self.clap_duration = 0

    def move_LRC(self, direction):
        """Di chuyển sang trái, phải hoặc trở về giữa."""
        if direction == "L" and self.x_position > self.LEFT:
            pyautogui.press('left')
            self.x_position -= 1
        elif direction == "R" and self.x_position < self.RIGHT:
            pyautogui.press('right')
            self.x_position += 1
        elif direction == "C" and self.x_position != self.CENTER:
            # Chuyển về giữa: nếu đang ở bên trái thì bấm phải, nếu ở bên phải thì bấm trái
            pyautogui.press('right' if self.x_position == self.LEFT else 'left')
            self.x_position = self.CENTER

    def move_JSD(self, action):
        """Điều khiển hành động nhảy, cúi xuống hoặc trở về trạng thái đứng."""
        if action == "J" and self.y_position == self.STAND:
            pyautogui.press('up')
            self.y_position = self.JUMP
        elif action == "D" and self.y_position == self.STAND:
            pyautogui.press('down')
            self.y_position = self.DOWN
        elif action == "S" and self.y_position != self.STAND:
            self.y_position = self.STAND

    def reset_game(self, image, results):
        """Reset game khi nhận đủ số frame vỗ tay."""
        # Nếu game đã bắt đầu thì reset trạng thái, ngược lại bắt đầu game mới
        if self.game_started:
            self.x_position = self.CENTER
            self.y_position = self.STAND
            pyautogui.press('enter')
        else:
            self.game_started = True
            # Có thể kích hoạt click chuột nếu cần
            # pyautogui.click(x=720, y=560, button="left")
        self.pose.save_shoulder_line_y(image, results)
        self.clap_duration = 0

    def play(self):
        """Vòng lặp chính của game."""
        cap = cv2.VideoCapture(0)

        while True:
            ret, image = cap.read()
            if not ret:
                continue

            image = cv2.flip(image, 1)
            image_height, image_width, _ = image.shape
            image, results = self.pose.detectPose(image)

            if results.pose_landmarks:
                if self.game_started:
                    # Kiểm tra chuyển động trái/giữa/phải
                    image, LRC = self.pose.checkPose_LRC(image, results)
                    self.move_LRC(LRC)

                    # Kiểm tra hành động nhảy/cúi/đứng
                    image, JSD = self.pose.checkPose_JSD(image, results)
                    self.move_JSD(JSD)
                else:
                    cv2.putText(image, "Clap your hand to start!",
                                (5, image_height - 10),
                                cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 0), 3)

                # Kiểm tra hành động vỗ tay
                image, CLAP = self.pose.checkPose_Clap(image, results)
                if CLAP == "C":
                    self.clap_duration += 1
                    if self.clap_duration == self.CLAP_THRESHOLD:
                        self.reset_game(image, results)
                else:
                    self.clap_duration = 0

            cv2.imshow("Game", image)
            if cv2.waitKey(1) == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    MyGame().play()

import cv2
import mediapipe as mp
import pyautogui
import numpy as np

class handDetector:
    def __init__(self, mode=False, maxHands=2, detectionCon=0.7, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = int(detectionCon)  # Convert to int
        self.trackCon = int(trackCon)          # Convert to int
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands,
                                        self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms,
                                               self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo=0, draw=True):
        lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
        return lmList

def main():
    cap = cv2.VideoCapture(0)
    detector = handDetector(maxHands=1, detectionCon=0.85, trackCon=0.8)

    screenWidth, screenHeight = pyautogui.size()
    smoothening = 5
    plocX, plocY = 0, 0
    clocX, clocY = 0, 0

    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        lmList = detector.findPosition(img, draw=False)

        if lmList:
            x1, y1 = lmList[8][1:]  # Tip of the index finger

            # Convert coordinates
            x3 = np.interp(x1, (100, 640-100), (0, screenWidth))
            y3 = np.interp(y1, (100, 480-100), (0, screenHeight))

            # Smoothen values
            clocX = plocX + (x3 - plocX) / smoothening
            clocY = plocY + (y3 - plocY) / smoothening

            # Debugging statements
            print(f'Index Finger Coordinates: ({x1}, {y1})')
            print(f'Mapped Coordinates: ({clocX}, {clocY})')

            # Move mouse
            pyautogui.moveTo(screenWidth - clocX, clocY)
            plocX, plocY = clocX, clocY

        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

import cv2
import mediapipe as mp
import time
class handdetector():
    def __init__(self,mode=False,max_hands=2,mc=1,det_con=0.5,tr_con=0.5):
        self.mode=mode
        self.max_hands=max_hands
        self.mc=mc
        self.det_con=det_con
        self.tr_con=tr_con
        self.mphand = mp.solutions.hands
        self.hands = self.mphand.Hands(self.mode,self.max_hands,self.mc,self.det_con,self.tr_con)
        self.mpdraw = mp.solutions.drawing_utils
    def find_hands(self,img,draw=True):
        imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        self.results=self.hands.process(imgRGB)
        #print(results.multi_hand_landmarks)
        if self.results.multi_hand_landmarks:
            for h in self.results.multi_hand_landmarks:
                if draw:
                    self.mpdraw.draw_landmarks(img,h,self.mphand.HAND_CONNECTIONS)
                # for id,lm in enumerate(h.landmark):
                #     h,w,c=img.shape
                #     cx,cy=int(lm.x*w),int(lm.y*h)
                #     print(id,cx,cy)
                #     cv2.putText(img, str(id), (cx, cy), cv2.FONT_HERSHEY_COMPLEX,0.75, (255, 0, 0), 1)
                    # if(id==3):
                    #     cv2.circle(img,(cx,cy),25,(255,0,255),cv2.FILLED)
        return img

    def find_pos(self,img,handno=0,draw=True):
        lmlist=[]
        if self.results.multi_hand_landmarks:
            myhand=self.results.multi_hand_landmarks[handno]
            for id,lm in enumerate(myhand.landmark):
                h,w,c=img.shape
                cx,cy=int(lm.x*w),int(lm.y*h)
                #print(id,cx,cy)
                lmlist.append([id,cx,cy])
                if draw:
                    cv2.putText(img, str(id), (cx, cy), cv2.FONT_HERSHEY_COMPLEX,0.75, (255, 0, 0), 1)
        return lmlist


    def main():
        cap = cv2.VideoCapture(0)
        ptime = 0
        detector=handdetector()
        while (True):
            success, img = cap.read()
            img=detector.find_hands(img)
            ctime = time.time()
            fps = 1 / (ctime - ptime)
            ptime = ctime
            cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_COMPLEX, 3, (255, 0, 255), 2)
            lmlist=detector.find_pos(img)
            print(lmlist)

            cv2.imshow("image", img)
            cv2.waitKey(1)


if __name__=="__main__":
    handdetector.main()
import cv2
import mediapipe as mp
class PoseEstimator:
    def __init__(self,mode=False,c=1,sl=True,es=False,ss=True,min1=0.5,min2=0.5):
        self.mode=mode
        self.c=1
        self.sl=sl
        self.es=es
        self.ss=ss
        self.min1=min1
        self.min2=min2

        self.mppose=mp.solutions.pose
        self.mpdraw=mp.solutions.drawing_utils
        self.pose= self.mppose.Pose(self.mode,
        self.c,
        self.sl,
        self.es,
        self.ss,
        self.min1,
        self.min2)
    def findpose(self,img,draw=True):
        imgrgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        self.results=self.pose.process(imgrgb)
        if(self.results.pose_landmarks):
            #print(self.results.pose_landmarks)
            if(draw):
                self.mpdraw.draw_landmarks(img,self.results.pose_landmarks,self.mppose.POSE_CONNECTIONS)
        return img
    def get_position(self,img,draw=False):
        if (self.results.pose_landmarks):
            lmlist=[]
            for id,lm in enumerate(self.results.pose_landmarks.landmark):
                h,w,c=img.shape
                # print(id,lm)
                cx=int(lm.x*w)
                cy=int(lm.y*h)
                lmlist.append([id,cx,cy])
                if draw:
                    cv2.circle(img,(cx,cy),5,(255,0,0),cv2.FILLED)
            return lmlist



    def main():
        cap = cv2.VideoCapture('Videos/exercise2.mp4')
        detector=PoseEstimator()
        while (True):
            success, img = cap.read()
            img=detector.findpose(img)
            lmlist=detector.get_position(img)
            print(lmlist)
            cv2.imshow("Image", img)

            cv2.waitKey(1)



if __name__=='__main__':
    PoseEstimator.main()
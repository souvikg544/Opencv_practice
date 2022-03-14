import cv2
import mediapipe as mp
import time

class FaceMesh:
    def __init__(self,static_image_mode=False,max_num_faces=2,refine_landmarks=False,min_detection_confidence=0.5,min_tracking_confidence=0.5):
        self.static_image_mode=static_image_mode
        self.max_num_faces=max_num_faces
        self.refine_landmarks=refine_landmarks
        self.min_detection_confidence=min_detection_confidence
        self.min_tracking_confidence=min_tracking_confidence
        self.mpdraw=mp.solutions.drawing_utils
        self.mpfm=mp.solutions.face_mesh
        self.facemesh=self.mpfm.FaceMesh(self.static_image_mode,self.max_num_faces,
                                         self.refine_landmarks,self.min_detection_confidence,
                                         self.min_tracking_confidence)

    def get_mesh(self,img):
        imgrgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        results=self.facemesh.process(imgrgb)
        if results.multi_face_landmarks:
            for flm in results.multi_face_landmarks:
                #self.mpdraw.draw_landmarks(img,flm,self.mpfm.FACEMESH_CONTOURS)
                lmlist=[]
                for id,lm in enumerate(flm.landmark):
                    h,w,c=img.shape
                    cx=int(lm.x*w)
                    cy=int(lm.y*h)
                    lmlist.append([id,cx,cy])
                    #cv2.putText(img,str(id),(cx,cy),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,255,0),1)

    def main():
        cap=cv2.VideoCapture('Videos/couple.mp4')
        ptime=0
        detector=FaceMesh()
        while(True):
            s,img=cap.read()
            detector.get_mesh(img)




            ctime=time.time()
            fps=1/(ctime-ptime)
            ptime=ctime

            cv2.putText(img,f'FPS:{int(fps)}',(20,70),cv2.FONT_HERSHEY_COMPLEX_SMALL,3,(0,255,0),3)
            cv2.imshow("Image",img)
            cv2.waitKey(1)


if __name__=='__main__':
    FaceMesh.main()


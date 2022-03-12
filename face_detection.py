import cv2
import mediapipe as mp
class FaceDetection:
    def __init__(self,s=1):
        self.s=s
        self.mpfd=mp.solutions.face_detection
        self.mpdraw=mp.solutions.drawing_utils
        self.fd=self.mpfd.FaceDetection()
    def showface(self,img,draw=True):
        imgrgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        self.results=self.fd.process(imgrgb)
        bboxs=[]
        if self.results.detections:
            for id,lm in enumerate(self.results.detections):
                #print(id,lm)
                #print(lm.location_data.relative_bounding_box)
                #self.mpdraw.draw_detection(img,lm)
                bboxc=lm.location_data.relative_bounding_box
                h,w,c=img.shape
                bbox=int(bboxc.xmin*w),int(bboxc.ymin*h),\
                    int(bboxc.width *w),int(bboxc.height*h)
                bboxs.append([id,bbox,lm.score[0]])
                cv2.rectangle(img,bbox,(255,0,255),2)
                cv2.putText(img,f'confidence : {int(lm.score[0]*100)}',(bbox[0],bbox[1]-20),
                            cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(255,0,255),1)
        return bboxs



    def main():
        #"Videos/exercise2.mp4"
        cap=cv2.VideoCapture(0)
        detector=FaceDetection()
        while(True):
            sucess,img=cap.read()
            bboxs=detector.showface(img)
            print(bboxs)

            cv2.waitKey(1)
            cv2.imshow("Image",img)


if __name__=='__main__':
    FaceDetection.main()

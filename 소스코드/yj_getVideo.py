import cv2              # for using OpenCV4.5 and CUDNN.
import copy
import numpy as np      # for making variety zeros array.
from csi_camera import CSI_Camera

# 4:3 Resolution (Capture Resolution: 4032 x 3040)
DISPLAY_WIDTH   = 1280
DISPLAY_HEIGHT  = 960
UPPER_NOT_ROI   = DISPLAY_HEIGHT // 4
LOWER_NOT_ROI   = 3 * DISPLAY_HEIGHT // 4

# Standard Lines
UPPER_STANDARD_LINE = 0
LOWER_STANDARD_LINE = 0
UPPER_NORMAL_LINE   = 0
LOWER_NORMAL_LINE   = 0

# Trackbar
def onChange(a):
    pass

        
if __name__ == "__main__":
    camera = CSI_Camera()
    camera.create_gstreamer_pipeline(
        sensor_id       = 0,
        sensor_mode     = 0,
        framerate       = 30,
        flip_method     = 2,
        display_height  = DISPLAY_HEIGHT,   
        display_width   = DISPLAY_WIDTH
    )
    camera.open(camera.gstreamer_pipeline)
    camera.start()
    cv2.namedWindow("DISPLAY", cv2.WINDOW_AUTOSIZE)
    
    cv2.createTrackbar("rowROI", "DISPLAY", 0, LOWER_NOT_ROI, onChange)
    cv2.createTrackbar("colROI", "DISPLAY", 0, 200, onChange)
    cv2.setTrackbarPos("rowROI", "DISPLAY", 0)
    cv2.setTrackbarPos("colROI", "DISPLAY", 100)
    
    out = cv2.VideoWriter('stickerVideo.mp4', cv2.VideoWriter_fourcc(*'XVID'), 10.0, (DISPLAY_WIDTH,DISPLAY_HEIGHT),True)
    
    if not camera.video_capture.isOpened():
        print("Unable to open camera")
        SystemExit(0)
    try:
        cnt = 0
        setFlag = False
        upper_lines = []
        lower_lines = []
        camera.start_counting_fps()
        gpu_frame = cv2.cuda_GpuMat()
        
        while cv2.getWindowProperty("DISPLAY", 0) >= 0:
            # Read frame and initialze.
            _, img = camera.read()
            rowpos = cv2.getTrackbarPos("rowROI", "DISPLAY")
            colpos = cv2.getTrackbarPos("colROI", "DISPLAY")
            img[:UPPER_NOT_ROI + rowpos, :] = img[LOWER_NOT_ROI + rowpos:,:] = 0
            img[:, :colpos] = img[:, DISPLAY_WIDTH - colpos:] = 0
            
            cv2.putText(img, "FPS: " + str(camera.last_frames_displayed), (20,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_4)
            out.write(img)
            cv2.imshow("DISPLAY", img)
            camera.frames_displayed += 1
            if cv2.waitKey(25) & 0xFF == 27: break
            
    finally:
        out.release()
        camera.stop()
        camera.release()
        cv2.destroyAllWindows()

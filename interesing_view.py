#interesing_view.py
import cv2
import time
import pygame
import imutils
import argparse
import datetime
import random
import numpy as np
from interesion_model import exciting
from pygame.locals import *
from sys import exit

# 创建参数解析器并解析参数
ap = argparse.ArgumentParser()

ap.add_argument("-i", "--image", help="path to the image file")
ap.add_argument("-v", "--video", help="path to the video file")
ap.add_argument("-n", "--name", type = str, default="Capture",help="window name")
ap.add_argument("-w", "--width", type = int,default=800,help="window width")
ap.add_argument("-ht", "--height", type = int,default=1000,help="window height")
ap.add_argument("-a", "--min-area", type=int, default=500, help="minimum area size")

args = vars(ap.parse_args())
# 如果video参数为None，那么我们从摄像头读取数据
if args.get("video", None) is None and args["image"] is None:
    camera = cv2.VideoCapture(0)
    #等待0.25秒
    time.sleep(0.25)
 
# 否则我们读取一个视频文件
else:
    camera = cv2.VideoCapture(args["video"])

width = args["width"]
height = args["height"]
frames = 0 #帧计数器
history = 20 #背景建模样本量
faceslen=0
#firstFrame = None
backgrouds = []
pedestrians = {} #行人字典

pygame.init()
pygame.display.set_caption('Interesing')
screen = pygame.display.set_mode((width+200,height-399),pygame.RESIZABLE)
screen.fill([0,0,0])#用黑色填充窗口
et=exciting(camera,history)
facearray=et.read_images_array()

backgrouds = et.readBackgroud(random.randint(0,19))
print(backgrouds)
if backgrouds != None:
	frames = 20

faces = [['陈思羽','18']]
faceShow = []

faceID = 0
f=0
#img= cv2.imread('face/face_gray/',cv2.IMREAD_GRAYSCALE)
while camera.isOpened():

	time = datetime.datetime.now().strftime(u"%Y-%d %I:%M:%S")
	(ret,cv_img)= camera.read()
	frame = imutils.resize(cv_img,width,height)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	#opencv的色彩空间是BGR，pygame的色彩空间是RGB
	img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

	et.show_text(screen,(10,frame.shape[0]-40),time,(0,131,195),True,30)
	pygame.display.update()

	pixl_arr = np.swapaxes(img, 0, 1)
	new_surf = pygame.pixelcopy.make_surface(pixl_arr)
	#设置窗口背景
	screen.blit(new_surf, (0, 0))

	for event in pygame.event.get():
		if event.type == pygame.MOUSEBUTTONDOWN:
			pass
		if event.type == pygame.QUIT:
			camera.release()
			pygame.quit()
			exit()

	for i in range(0,len(faceShow)):
		
		roj = cv2.cvtColor(faceShow[i], cv2.COLOR_RGB2BGR)
		roj = np.swapaxes(roj, 0, 1)
		roj = pygame.pixelcopy.make_surface(roj)
		et.show_text(screen,(width-100,10),str(f),(251,116,135),30)
		et.show_text(screen,(width,i*200),str(i),(251,65,90),40)

		screen.blit(roj, (width, i*200))


	if frames < history:
		et.show_text(screen, (100,200),u"请离开镜头",(255, 255, 255), True,120)
		et.show_text(screen, (110,320),u"背      景      建      模      中:  {}%".format((frames/history)*100),
			(255, 255, 255), True,40)

		KNN=et.KNN_difference(frame,args["min_area"])
		rect = et.people(frame) #人检查
		dets = et.dlibFace(img)
		face=et.face(gray) #脸
		
		if rect != [] or face != () or dets != None or KNN != []:

			for x,y,w,h in rect:
				pygame.draw.rect(screen,[247,0,34],[x,y,w-x,h-y],3)
			for fx,fy,fw,fh in face:
				pygame.draw.rect(screen,[255,149,0],[fx,fy,fw,fh],3)
			for r in KNN:
				x,y,w,h = et.wrap_digit(r)
				pygame.draw.rect(screen,[106,243,62],[x,y,w,h],3)

			#for i,d in enumerate(dets):
				#pygame.draw.rect(screen,[163,0,22],[d.left(),d.top(),d.right(),d.bottom()],3)
			continue
		else:
			et.writeBackgroud(frame,frames)
			frames += 1
			continue
	
	#识别开始
	#差度
	#backgrouds=et.readBackgroud(random.randint(0,19))
	#contours=et.frame_difference(backgrouds,gray)
	#KNN
	KNN=et.KNN_difference(frame,args["min_area"])
	if  KNN != []:

		#rect = et.people(frame) #人检查
		face=et.face(gray) #脸

		if face != ():
			for fx,fy,fw,fh in face:
				pygame.draw.rect(screen,[255,149,0],[fx,fy,fw,fh],3)
				roi = gray[fy:fy+fh,fx:fx+fw]
				roj = frame[fy:fy+fh,fx:fx+fw]
				roi = cv2.resize(roi,(200,200))
				roj = cv2.resize(roj,(200,200))

				faceShow.append(roj)
				if facearray != []:

					x.append(np.asarray(roi,dtype=np.uint8))
					y.append(faceslen)
					et.face_rec([x,y])
					cv2.imwrite('face/face_gray/1/%s.png' % str(faceID),roi)
					cv2.imwrite('face/face_color/1/%s.png' % str(faceID),roj)
					faceslen = faceslen+1


				et.face2(roj)

				if f<20:
					cv2.imwrite('face/face_gray/1/%s.png' % str(f),roi)
					cv2.imwrite('face/face_color/1/%s.png' % str(f),roj)

					
					f =f+1
					
					
				#screen.blit(roi, (0, 0))


				#faceName=et.face2(roi)
				#print(faceName)
				for i in range(0,len(faces)):
					for j in range(0,len(faces[i])):
						pygame.draw.rect(screen,[193,133,47],[fx+fw,fy+(42*j),90,40])
						et.show_text(screen,(fx+fw,fy+(45*j)),faces[i][j],(255,255,255), True,30)
		
	
		
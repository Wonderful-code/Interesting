#interesion_model.py
import cv2
import dlib
import time
import datetime
import imutils
import numpy as np 
import pygame
from imutils.object_detection import non_max_suppression

class exciting(object):
	"""docstring for """
	def __init__(self,camera,history):

		#初始化方向梯度直方图描述子/设置支持向量机
		self.hog = cv2.HOGDescriptor()
		self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
		#初始化背景分割器
		self.bg = cv2.createBackgroundSubtractorKNN(detectShadows=True)
		self.bg.setHistory(history)
		self.detector = dlib.get_frontal_face_detector()
		self.face_cascade = cv2.CascadeClassifier('haarcascades//haarcascade_frontalface_default.xml')

		self.model = None
		self._camera = camera
		self._videoWriter = None
		self._videoFilename = None
		self._framesElapsed = None
		self._fpsEstimate = None
		self._startTime = None
		self._videoEncoding = None
		


	def inside(self,r1,r2):
		x1,y1,w1,h1 = r1
		x2,y2,w2,h2 = r2
		if (x1>x2) and (y1>y2) and (x1+w1<x2+w2) and (y1+h1<y2+h2):
			return True

		else:
			return False 

	def show_text(self,screen, pos, text, color, font_bold = False, font_size = 50, font_italic = False):   
		'''
		Function:文字处理函数 
		Input:screen：surface句柄
				pos：文字显示位置 
				color:文字颜色 
				font_bold:是否加粗 
				font_size:字体大小 
				font_italic:是否斜体 
		Output: NONE '''
		cur_font = pygame.font.SysFont("SourceHanSans-Bold", font_size)
		cur_font.set_bold(font_bold)  
		cur_font.set_italic(font_italic)  
		text_fmt = cur_font.render(text, 1, color)  
		return screen.blit(text_fmt, pos)

	def wrap_digit(self,rect):
		x,y,w,h = rect
		padding = 5
		hcenter = x+w/2
		vcenter = y+w/2
		if(h>w):
			w = h
			x = hcenter - (w/2)
		else:
			h = w
			y = vcenter - (w/2)
		return (x-padding,y-padding,w+padding,h+padding)


	def readBackgroud(self,count):
		return cv2.imread('Background//%s.png' % str(count),cv2.IMREAD_COLOR)

	def writeBackgroud(self,frame,count):
		#(帧，张)
		return cv2.imwrite('Background/%s.png' % str(count),frame)

	def writeImg(self,frame,path,count):
		return cv2.imwrite(path+'/%s.png' % str(count),frame)

	def read_images_array(self,path='face/face_gray/',l=20,z=0):
	#读取图片（路径，张数，人数，图片太大了）
	#face/str(1)/str(i)+'png'
		c=0
		x,y=[],[]
		for o in range(0,z+1):
			for i in range(0,l+1):
				try:
					path_in = path +str(o+1) +'/'+ str(i) + '.png'
					im = cv2.imread(path_in,cv2.IMREAD_GRAYSCALE)
					im = imutils.resize(im,32,32)
				#im = cv2.resize(im,(100,100),interpolation = cv2.INTER_LINEAR)
				
					x.append(np.asarray(im,dtype=np.uint8))
					y.append(c)
					path_in = None
				except:
					continue
			c=c+1
		return [x,y]
	@property
	def isFirstFace(self):
		return self._firstFace is not True

	def face_rec(self,array):
		try:
			[x,y] = array
			y=np.asarray(y,dtype=np.int32)
			self.model = cv2.face.createEigenFaceRecognizer()
			self.model.train(np.asarray(x),np.asarray(y))
			
		except:
			return False

	def face2(self,roi):
		try:
			roi = cv2.resize(roi,(32,32),interpolation = cv2.INTER_LINEAR)
			return self.model.predict(roi)
			
		except:
			return

	def frame_difference(self,firstFrame,gray):

		avg = cv2.cvtColor(firstFrame,cv2.COLOR_BGR2GRAY)

		gray = cv2.GaussianBlur(gray,(21,21),0)
		avg = cv2.GaussianBlur(avg,(21,21),0)

		#cv2.accumulateWeighted(gray,avg,0.5)
		frameDelta =cv2.absdiff(gray, cv2.convertScaleAbs(avg))
		thresh =cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
		# 扩展阀值图像填充孔洞，然后找到阀值图像上的轮廓
		thresh =cv2.dilate(thresh, None, iterations=2)
		(_,cnts, _) =cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

		return cnts

	def KNN_difference(self,frame,min_area=500):

		fgmask = self.bg.apply(frame)
		thresh = cv2.threshold(fgmask, 25, 255, cv2.THRESH_BINARY)[1]
	
		thresh = cv2.dilate(thresh, None, iterations=2)
		#findContours(image, mode, method[, contours[, hierarchy[, offset]]]) -> image, contours, hierarchy
		(_,cnts,_)= cv2.findContours(thresh.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

		rectangles = []

		for c in cnts:
			if cv2.contourArea(c) < min_area:
				continue
			r = (x, y, w, h) = cv2.boundingRect(c)

			is_inside = False
			for q in rectangles:
				if self.inside(r,q):
					is_inside = True
					break		
			if not is_inside:
				rectangles.append(r)

		return rectangles


	def people(self,frame):
		#人
		#调整到（1）减少检测时间，将图像裁剪到最大宽度为400个像素
		frame = imutils.resize(frame, width=min(400, frame.shape[1]))
		#检测图像中的人
		(rects, weights) = self.hog.detectMultiScale(frame, winStride=(4, 4),
			padding=(8, 8), scale=1.05)
		# 应用非极大值抑制的边界框
		# 相当大的重叠阈值尽量保持重叠
		rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
		return non_max_suppression(rects, probs=None, overlapThresh=0.65)

	def face(self,gray):
		#脸
		return self.face_cascade.detectMultiScale(gray, 1.3, 5)

	def dlibFace(self,frame):
		l = None
		dets = self.detector(frame, 0)
		for i,d in enumerate(dets):
			#pygame.draw.rect(screen,[163,0,22],[d.left(),d.top(),d.right(),d.bottom()],3)
			l = d.left()
		return l 
	def car(self):
		#汽车
		pass

	def Ev(self):
		#电动车
		pass

	def bicycle(self):
		#自行车
		pass

	def Monitor(self,path,frame,encoding = cv2.VideoWriter_fourcc('I','4','2','0')):
		#更新FPS估计和相关变量。
		if self._framesElapsed == 0:
			self._startTime = time.time()
		else:
			timeElapsed = time.time() - self._startTime
			self._fpsEstimate = self._framesElapsed/timeElapsed
		self._framesElapsed += 1

		self._videoFilename = path
		self._videoEncoding = encoding

		self._writeVideoFrame()

	def _writeVideoFrame(self):
			
		if not self.isWritingVideo:
			return
		if self._videoWriter is None:
			fps = self._camera.get(cv2.CAP_PROP_FPS)
			if fps == 0.0:
				if self._framesElapsed <20:
					return
				else:
					fps = self._fpsEstimate
			size = (int(self._camera.get(cv2.CAP_PROP_FRAME_WIDTH)),int(self._camera.get(cv2.CAP_PROP_FRAME_HEIGHT)))

			self._videoWriter = cv2.VideoWriter(self._videoFilename,self._videoEncoding,fps,size)
		self._videoWriter.Write(self._frame)

	@property #在写视频吗？
	def isWritingVideo(self):
		return self._videoFilename is not None
#interesion_model.py
'''
*******************************************************************************************************
*******************************************************************************************************
**                                                                                                   **
**                                                                                                   **
**     IIIIII  NN   NN  TTTTTT   EEEEEEE   RRRRR    EEEEEE    SSSSS   IIIIII    OOOO    NN   NN      **
**       II    NNN  NN    TT    EE        RR   RR  EE        SS   SS    II    OO    OO  NNN  NN      **
**       II    NNNN NN    TT    EEEEEEEE  RR   RR  EEEEEEE    SS        II    OO    OO  NNNN NN      **
**       II    NN NNNN    TT    EEEEEEEE  RRRRR    EEEEEEE      SS      II    OO    OO  NN NNNN      **
**       II  　NN  NNN    TT    EE        RR  RR   EE        SS   SS    II    OO    OO  NN  NNN      **
**     IIIIII　NN   NN    TT     EEEEEEE  RR   RR   EEEEEE    SSSSS   IIIIII    OOOO    NN   NN      **
**                                                                                                   **
**                                                                                                   **
*******************************************************************************************************
*******************************************************************************************************
Interesion 是一款监控,脸部识别
interesion_model 是Interesion 的核心模块

备忘录：　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　
面人脸分类器进行了实验，总共有4个，alt、alt2、alt_tree、default。
对比下来发现alt和alt2的效果比较好，alt_tree耗时较长，default是一个轻量级的，
经常出现误检测。所以还是推荐大家使用haarcascade_frontalface_atl.xml和
haarcascade_frontalface_atl2.xml。

'''
import os
import sys
import cv2
import dlib
import time
import json
import serial
import random
import pygame
import imutils
import datetime
import numpy as np
from multiprocessing import Process
from pygame_model import pygameDraw
from imutils.object_detection import non_max_suppression

class exciting(object):

	def __init__(self,camera,#摄像头对象
					pygameDraw,
					name='Interesing',
					width =800,
					height=1000,
					show = False,
					history=20,#背景建模样本量
					args=500):#忽略大小

		self.args = args
		self.name = name
		self.width = width
		self.height = height
		self.camera = camera
		self.history = history

		self.gray = None
		self.model = None
		self._frame = None
		self._color = None
		self._params = None
		self._arduino = None
		self._startTime = None
		self._backgrouds = None
		self._fpsEstimate = None
		self._videoWriter = None
		self._videoFilename = None
		self._videoEncoding = None
		self._facearray=None
		self._isc2show = False

		self._faceN = 0
		self._frames = 0
		self._bgframes = 0 #帧计数器
		self._firstFace = 0
		self._faceTooMuch = 10000
		self._framesElapsed = 0
		
		self._val= 4 #初始的摄像头角度
		self._rinit = 4 #脸部丢失时的回归角度
		self._space = 5 #多少次脸部检查后的摄像头调整
		self._timeSpace = 0 #识别到脸部次数

		self._data=[]
		self._faceShow=[]
		self._frameImg=[]
		self.detector = dlib.get_frontal_face_detector()
		self.predictor=dlib.shape_predictor("haarcascades//shape_predictor_68_face_landmarks.dat") 
		#self.hog = cv2.HOGDescriptor()#初始化方向梯度直方图描述子/设置支持向量机
		#self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
		self.bg = cv2.createBackgroundSubtractorKNN(detectShadows=True)#初始化背景分割器
		self.bg.setHistory(self.history)
		self.face_alt2 = cv2.CascadeClassifier('haarcascades//face.xml')
		self.path = ['Background','face','face/face_color','face/face_gray','faceImg']

		self.pd = pygameDraw
		self.imgflie

	def inside(self,r1,r2):
		x1,y1,w1,h1 = r1
		x2,y2,w2,h2 = r2
		if (x1>x2) and (y1>y2) and (x1+w1<x2+w2) and (y1+h1<y2+h2):
			return True
		else:
			return False 

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

	@property #在写视频吗？
	def isWritingVideo(self):
		return self._videoFilename is not None
	@property
	def Img(self):
		self.loadJson
		path_1 = self.path[4]+'/'+self.path[5]
		for parent,dirnames,filenames in os.walk(path_1):#三个参数：分别返回1.父目录 2.所有文件夹名字（不含路径） 3.所有文件名字
			for filename in filenames: #输出文件信息
				path = os.path.join(parent,filename)
				Img1 = pygame.image.load(path).convert_alpha()
				self._frameImg.append(Img1)

	@property
	def loadJson(self):
		try:
			with open('face.json', 'r',encoding='UTF-8') as f:
				self._data=json.load(f)
				img = self._data["data"]
				self._data=self._data[img]
				self.path.append(self._data[0])
		except:
			pass
	@property
	def FPS(self):
		#更新FPS估计和相关变量。
		if self._framesElapsed == 0:
			self._startTime = time.time()
		else:
			timeElapsed = time.time() - self._startTime
			self._fpsEstimate = self._framesElapsed/timeElapsed
		self._framesElapsed += 1
		self._frames += 1
	#保存脸部图片
	def facePng(self,color,gray):
		try:
			self._faceN += 1
			path_gray = self.path[3]  +'/'+ str(self._faceN) + '.png'
			path_color = self.path[2] +'/'+ str(self._faceN) + '.png'
			cv2.imwrite(path_gray,gray)
			cv2.imwrite(path_color,color)
		except:
			print("保存脸失败！")
	#文件夹是否存在，不在创建
	@property
	def imgflie(self):
		for path in self.path:
			if os.path.exists(path):
				print("OK")
			else:
				os.mkdir(path)
	@property
	def backgrouds():#随机读取一张背景图片，如果读取到了就不用背景建模
		self._backgrouds = self.readBackgroud(random.randint(0,19))
		if self._backgrouds != None:
			self._bgframes = 20
	def readBackgroud(self,count):
		return cv2.imread('Background//%s.png' % str(count),cv2.IMREAD_COLOR)

	def writeBackgroud(self,count):#(帧，张)
		return cv2.imwrite('Background//%s.png' % str(count),self._frame)

	def writeImg(self,frame,path,count):
		return cv2.imwrite(path+'/%s.png' % str(count),frame)
	#内部类把帧写入视频文件
	def _writeVideoFrame(self):
		if not self.isWritingVideo:
			return
		if self._videoWriter is None:
			fps = self.camera.get(cv2.CAP_PROP_FPS)
			if fps == 0.0:
				if self._framesElapsed <20:
					return
				else:
					fps = self._fpsEstimate
			size = (int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH)),int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT)))
			self._videoWriter = cv2.VideoWriter(self._videoFilename,self._videoEncoding,fps,size)
		self._videoWriter.Write(self._frame)
	#开始录像
	def Monitor(self,path,frame,encoding = cv2.VideoWriter_fourcc('I','4','2','0')):
		self._videoFilename = path
		self._videoEncoding = encoding
		self._writeVideoFrame()
	#识别身份用（暂时没用）	
	def confirm(self,face,faces):#face 区域，faces 
		for i in range(0,len(faces)):
			for j in range(0,len(faces[i])):
				self.pd.drawFace(face,True)
				self.pd.drawConfirm(self,x,y,w,c=False)
	#基础脸部识别		
	def face(self):
		return self.face_alt2.detectMultiScale(self.gray, 1.3, 5)#脸
	#dlib 脸识别器 画图片到脸上绘制
	def Img_to_Face(self,show = False):
		if self._frames > int(self._data[1]):
			self._frames = 0
		dets = self.detector(self._color, 0)
		for i,d in enumerate(dets):
			try:
				shape = self.predictor(self.gray,d)
				if show:
					for i in range(68):
						pt=shape.part(i)
						fa=shape.part(30)
						x = fa.x+int(self._data[2])-85
						y = fa.y+int(self._data[3])-30
						self.pd._screen.blit(self._frameImg[self._frames], (x,y))

						#self.pd.show_text((pt.x,pt.y),str(i),15,True,10)
						#self.pd.circle(pt.x,pt.y)	
			except:
				pass

	#返回diib识别对象
	def dlibdate(self):
		return self.detector(self._frame, 0)
	#两帧不同
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
	#KNN（运动跟踪）
	def KNN_difference(self,frame,min_area=500):
		fgmask = self.bg.apply(frame)
		thresh = cv2.threshold(fgmask, 25, 255, cv2.THRESH_BINARY)[1]
		thresh = cv2.dilate(thresh, None, iterations=2)
		#findContours(image, mode, method[, contours[, hierarchy[, offset]]]) -> image, contours, hierarchy
		(_,cnts,_)= cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
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
		self._KNN = rectangles
	#人识别（有点慢）
	def people(self,frame):
		#调整到（1）减少检测时间，将图像裁剪到最大宽度为400个像素
		frame = imutils.resize(frame, width=min(400, frame.shape[1]))
		#检测图像中的人
		(rects, weights) = self.hog.detectMultiScale(frame, winStride=(4, 4),
			padding=(8, 8), scale=1.05)
		# 应用非极大值抑制的边界框
		# 相当大的重叠阈值尽量保持重叠
		rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
		return non_max_suppression(rects, probs=None, overlapThresh=0.65)
	#背景建模
	@property
	def bgbuild(self):
		try:
			back = self.readBackgroud(5)
		except:
			if self._bgframes < self.history:
				if self._isc2show:
					pass
				else:
					self.pd.show_text((100,200),u"请离开镜头",1,True,120)
					self.pd.show_text((110,320),u"背      景      建      模      中:  {}%".format((self._bgframes/self.history)*100),1,True,40)

				KNN=self.KNN_difference(self._frame,self.args)
				rect = self.people(self._frame) #人检查
				l,t,w,h= self.dlibFace()
				face = self.face(self.gray) #脸
			
				if rect != [] or face != () or l != None or KNN != []:
					self.pd.drawKNN(KNN)
					self.pd.drawPeople(rect)
					self.pd.drawFace(face)
				else:
					self.writeBackgroud(self._bgframes)
					self._bgframes += 1
	#dlib识别脸部绘制	
	def discern(self):
		KNN=self.KNN_difference(self._frame,self.args)
		if  KNN != []:
			try:
				dets = self.dlibdate()
				for i,d in enumerate(dets):
					(x1,y1,x2,y2) = d.top(),d.bottom(),d.left(),d.right()
					face = self._color[x1:y1,x2:y2]
					face = imutils.resize(face,200,200) #设置大小
					self._faceShow.append(face)
					self.pd.drawFaces(self._faceShow)
					self.pd.drawFace(x1,y1,x2,y2)#显示区域
					self.pd.show_text((x2,x1-40),'index:%s' % str(i),14,True,30)
					if len(self._faceShow) > 10:
						self._faceShow.clear()
			except:
				pass

	#face识别器
	def discernFace(self,save = False):
		KNN=self.KNN_difference(self._frame,self.args)
		if  KNN != []:
			faces=self.face() #脸
			if faces != ():
				for x,y,w,h in faces:
					face = self._color[y:y+h,x:x+w]
					face = imutils.resize(face,200,200) #设置大小
					if save and self._faceN<self._faceTooMuch:
						color = self._frame[y:y+h,x:x+w]
						gray = self.gray[y:y+h,x:x+w]
						gray = imutils.resize(gray,200,200)
						color = imutils.resize(color,200,200)
						self.facePng(color,gray)
					self._faceShow.append(face)
					self.pd.drawFaces(self._faceShow)
					self.pd.drawFace2(x,y,w,h)#显示区域
					if len(self._faceShow) > 10:
						self._faceShow.clear() #超过10张 释放内存

	#无窗口 帧获取
	def NoWinStart(self):
		(ret,cv_img)= self.camera.read() #获得帧
		self._frame = imutils.resize(cv_img,self.width,self.height) #设置大小
		self.gray = cv2.cvtColor(self._frame, cv2.COLOR_BGR2GRAY)#灰色
	#窗口 帧获取
	@property
	def start(self):
		self.FPS		  #帧数率
		(ret,cv_img)= self.camera.read() #获得帧
		self._frame = imutils.resize(cv_img,self.width,self.height) #设置大小
		self.gray = cv2.cvtColor(self._frame, cv2.COLOR_BGR2GRAY)#灰色
		self._color = cv2.cvtColor(self._frame, cv2.COLOR_RGB2BGR)#opencv的色彩空间是BGR，pygame的色彩空间是RGB
		try:
			self.pd.show_text((10,10),'FPS:%.2f' % self._fpsEstimate,14,True,30)
			self.pd.show_text((10,self._frame.shape[0]-40),datetime.datetime.now().strftime(u"%Y-%d %I:%M:%S"),15,True,30)
			pygame.display.update()
			pixl_arr = np.swapaxes(self._color, 0, 1)
			new_surf = pygame.pixelcopy.make_surface(pixl_arr)
			self.pd._screen.blit(new_surf, (0, 0))
			self.pd.quit(self.camera) #pygame 事件
		except:
			pass
	#提供端口
	def ArduinoToPort(self,port):
		try:
			self._arduino = serial.Serial('com'+str(port),9600)
		except:
			print('端口没链接到Arduino')
			pass
	#Arduino 初始化
	def ArduinoInit(self,rinit = 4,space = 5):

		self._val= 4 #初始的摄像头角度
		self._rinit = rinit #脸部丢失时的回归角度
		self._space = space #多少次脸部检查后的摄像头调整
		self._timeSpace = 0 #识别到脸部次数
		#扫描20个端口
		for i in range(0,20):
			t = 0
			try:
				self._arduino = serial.Serial('com'+str(i),9600)
				print('链接到com'+str(i)+'端口')
				print('联系对方：Hello!')
				if t == 5:
					self._arduino.write('0'.encode())
					time.sleep(1)
					data = self._arduino.read(1)
					if data !='':
						print(data)
						if data == 'Y':
							print('is me')
						else:
							break
					t+=1
				else:
					continue
			except:
				print('com'+str(i)+'端口没链接到Arduino')
	#转动基础
	def toArduino(self,faceMode):
		x,y = 0,0
		face =()
		dets = None
		if self._timeSpace > 1000:
			self._timeSpace = 0
		else:
			self._timeSpace += 1
		try:
			KNN=self.KNN_difference(self._frame,self.args)
			if KNN != []:
				if face != []:
					self.pd.drawFace2(face)#显示区域
					for ax,ay,w,h in face:
						x,y = ax,ay
						print(x,y)
				elif dets != None:
					for i,d in enumerate(dets):
						x,y = d.left(),d.top()
				else:
					self._val = self._rinit
					i = str(self._val)
					self._arduino.write(i.encode())
					time.sleep(0.25)

				if faceMode == 'face':
					face=self.face()
				elif faceMode == 'dlib':
					dets=self.dlibdate()

				
			if self._val >= 9:
				self._val = 9
			elif self._val <=0:
				self._val = 0

			if x != 0:
				if x < 280 and (self._timeSpace%self._space) == 0:
					self._val += 1
					i = str(self._val)
					self._arduino.write(i.encode())
					time.sleep(0.25)
				elif x > 470 and (self._timeSpace%self._space) == 0:
					self._val -= 1
					i = str(self._val)
					self._arduino.write(i.encode())
					time.sleep(0.25)
		except:
			pass
	#带窗口
	def toWinArduino(self,faceMode='face'):#face or dlib
		self.start
		self.toArduino(faceMode)
	#不带窗口
	def NoWinToArduino(self,faceMode='face'):#face or dlib
		self.NoWinStart()
		self.toArduino(faceMode)

	



def car(self):
	#汽车
	pass

def Ev(self):
	#电动车
	pass

def bicycle(self):
	#自行车
	pass
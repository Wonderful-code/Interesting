#pygame_model.py
'''
*******************************************************************************************************
*******************************************************************************************************
**                                                                                                   **
**                        　　                                                                       **
**　　 IIIIII  NN   NN  TTTTTT   EEEEEEE   RRRRR    EEEEEE    SSSSS   IIIIII    OOOO    NN   NN      **
**　　   II    NNN  NN    TT    EE        RR   RR  EE        SS   SS    II    OO    OO  NNN  NN      **
** 　　　II　　NNNN NN    TT    EEEEEEEE  RR   RR  EEEEEEE    SS        II    OO    OO  NNNN NN      **
**       II    NN NNNN    TT    EEEEEEEE  RRRRR    EEEEEEE      SSS     II    OO    OO  NN NNNN      **
** 　　  II  　NN  NNN    TT    EE        RR  RR   EE        SS   SS    II    OO    OO  NN  NNN      **
** 　　IIIIII　NN   NN    TT     EEEEEEE  RR   RR   EEEEEE    SSSSS   IIIIII    OOOO    NN   NN      **
**                                                                                                   **
**                                                                                                   **
*******************************************************************************************************
*******************************************************************************************************
Interesion 是一款监控，脸部识别
pygame_model 是Interesion 的pygame 显示模块

1.pygame主模块
2.显示模块

备忘录：
主色:
#F70022       #BB2D41       #A30016 	#FB415A		   #FB7487
(247,0,34)  (187,45,65)    (163,0,22)  (251,65,90)   (251,116,135)
辅助色 A:
#FF9500		  #C1852F		#A86300	    #FFB142	       #FFC676
(255,149,0)  (193,133,47)  (168,99,0)  (255,177,66)  (255,198,118)
辅助色 B:
#0068C5		  #246095		#004582		#3A93E3		   #69A9E3
(0,104,197)  (36,96,149)   (0,69,130)  (58,147,227)   (105,169,227)
互补色:
#37E600		  #4AAE2A		#259800		#6AF33E		   #90F370
(55,230,0)  (74,174,42)   (37,152,0)  (106,243,62)   (106,243,62)
'''		
import pygame
import datetime
from sys import exit
from pygame.font import * 
from pygame.locals import *

class pygameDraw(object):
	def __init__(self,name,height,width):

		self._name = name
		self._height = height
		self._width = width
		self._screen = None
				      #1           2
		self._color = [[0,0,0],[255,255,255],
					  #新的脸3       4           5           6           7
					  [247,0,34],[187,45,65],[163,0,22],[251,65,90],[251,116,135],
					  #8             9            10          11           12
					  [255,149,0],[193,133,47],[168,99,0],[255,177,66],[255,198,118],
					  #13            14           15          16            17
					  [0,104,197],[36,96,149],[0,69,130],[58,147,227],[105,169,227],
					  #18            19          20           21          22
					  [55,230,0],[74,174,42],[37,152,0],[106,243,62],[106,243,62]]
		self.init()

	def init(self):
		pygame.init()
		pygame.display.set_caption(self._name)
		self._screen = pygame.display.set_mode((self._width,self._height-399),pygame.RESIZABLE)
		self._screen.fill(self._color[1])#用黑色填充窗口

	def newWind(self):
		self._screen = pygame.display.set_mode((self._width+200,self._height-399),pygame.RESIZABLE)
		self._screen.fill(self._color[1])#用黑色填充窗口

	def show_text(self, pos, text, color, font_bold = False, font_size = 13, font_italic = False):
		#Function:文字处理函数 pos：文字显示位置 color:文字颜色 font_bold:是否加粗 font_size:字体大小 font_italic:是否斜体 Output: NONE 
		cur_font = pygame.font.SysFont("SourceHanSans-Bold", font_size)#获取系统字体，并设置文字大小  
		cur_font.set_bold(font_bold)#设置是否加粗属性  
		cur_font.set_italic(font_italic)#设置是否斜体属性
		text_fmt = cur_font.render(text, 1, self._color[color])#设置文字内容 
		return self.screen.blit(text_fmt, pos)#绘制文字 
	@property
	def time(self):
		return self.show_text((10,frame.shape[0]-40),datetime.datetime.now().strftime(u"%Y-%d %I:%M:%S"),(0,131,195),True,30)
	def drawPeople(self,rect):
		for x,y,w,h in rect:
			pygame.draw.rect(self._screen,self._color[19],[x,y,w-x,h-y],3)
	def drawKNN(self,KNN):
		for r in KNN:
			x,y,w,h = self.wrap_digit(r)
			pygame.draw.rect(self._screen,self._color[18],[x,y,w,h],3)
	def drawFace(self,face,c=False):
		if c:
			color = 3 #新脸
		else:
			color = 4 #已识别
		for fx,fy,fw,fh in face:
			pygame.draw.rect(self._screen,self._color[color],[fx,fy,fw,fh],3)
	def drawConfirm(self,x,y,w,c=False):
		if c:
			color = 3 #新脸
		else:
			color = 4 #已识别
		pygame.draw.rect(et.screen,self.color[color],[x+w,y+(42*j),90,40])
	
	def drawFace(faceShow):
		for i in range(0,len(faceShow)):
			roj = cv2.cvtColor(faceShow[i], cv2.COLOR_RGB2BGR)
			roj = np.swapaxes(roj, 0, 1)
			roj = pygame.pixelcopy.make_surface(roj)
			self.show_text(et.screen,(width-100,10),str(f),(251,116,135),30)
			self.show_text(et.screen,(width,i*200),str(i),(251,65,90),40)

			self._screen.blit(roj, (width, i*200))

	def drawfames():
		self.time
		pygame.display.update()
		pixl_arr = np.swapaxes(self._color, 0, 1)
		new_surf = pygame.pixelcopy.make_surface(pixl_arr)
		
		self._screen.blit(new_surf, (0, 0))#设置窗口背景

'''
		视频: pygame.movie
要在游戏中播放片头动画、过场动画等视频画面，可以使用模块。

要播放视频中的音乐，pygame.movie模块需要对音频接口的完全控制，不能初始化mixer模块。因此要这样完成初始化

pygame.init()
pygame.mixer.quit()
或者只初始化 pygame.display.init()
movie = pygame.movie.Movie('filename') 指定文件名载入视频。视频的格式可以为mpeg1。视频文件不会马上全部载入内存，而是在播放的时候一点一点的载入内存。
movie.set_display(pygame.display.set_mode((640,480))) 指定播放的surface。
movie.set_volume(value) 指定播放的音量。音量的值value的取值范围为0.0到1.0。
movie.play() 播放视频。这个函数会立即返回，视频在后台播放。这个函数可以带一个参数loops，指定重复次数。 正在播放的视频可以用
movie.stop() 停止播放。
movie.pause() 暂停播放。
movie.skip(seconds) 使视频前进seconds秒钟。
NOTE：在某些XWindow下，可能需要配置环境变量： export SDL_VIDEO_YUV_HWACCEL=0
'''
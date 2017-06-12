# Interesting

Interesting

Interesting 是一个脸部识别的python软件，目前可以实现：
        1.脸部区域识别，区域画矩形
        2.获取脸部区域，保持color，gray俩个版本200*200的脸部图片
        3.显示识别到的脸部
        
 希望加入的功能：
        1.人脸身份识别
        2.加入tensorFolw实现人脸学习
        
 使用:
    用摄像头：
    
        'python interesing_view.py'
        
        "-i", "--image"，path to the image file
        " -v", "--video", path to the video file
        "-n", "--name", type = str, default="Capture"window name
        "-w", "--width", type = int,default=800,help="window width
        "-ht", "--height", type = int,default=1000,help="window height
        "-a", "--min-area", type=int, default=500, help="minimum area size

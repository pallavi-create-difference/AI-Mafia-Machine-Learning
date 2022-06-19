
import cv2
import numpy as np


face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eyes_cascade = cv2.CascadeClassifier("./Train/third-party/haarcascade_eye.xml")
nose_cascade = cv2.CascadeClassifier("./Train/third-party/haarcascade_mcs_nose.xml")

img = cv2.imread('./Test/Before.png')
mustache = cv2.imread('./Train/mustache.png',-1)
glasses = cv2.imread('./Train/glasses.png',-1)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)#make picture gray
 
faces = face_cascade.detectMultiScale(gray, 1.3, 5)

for (x,y,w,h) in faces:
	roi_gray = gray[y:y+h, x:x+w]
	roi_color = img[y:y+h, x:x+w]
	#cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2)


	eyes = eyes_cascade.detectMultiScale(roi_gray,1.3,5)
	el = []
	for(ex,ey,ew,eh) in eyes:
		#cv2.rectangle(roi_color,(ex, ey), (ex+ew, ey+eh),(0,255,0),3)
		el.append((ex,ey,ew,eh))
		roi_eyes = roi_gray[ey:ey+eh, ex:ex+w]
	
	el = sorted(el , key = lambda a : a[0])
	print(el)	
	ewf = el[1][0] + el[1][2] - el[0][0]
	ehf = el[1][1] + el[1][3] - el[0][1]

	print(ewf,ehf)
	glasses2 = cv2.resize(glasses.copy(),(int(1.2*ewf),int(2*ehf)))
	print(glasses2.shape)
	gw, gh, gc = glasses2.shape
	for i in range(0,gw):
		for j in range(0,gh):
			if glasses2[i,j][3] != 0:
				roi_color[int((el[0][1]+el[1][1])/3)+i, int((h-gh)/2)+j] = glasses2[i, j]



	nose = nose_cascade.detectMultiScale(roi_gray, scaleFactor=1.5, minNeighbors=5)
	for (nx, ny, nw, nh) in nose:
		#cv2.rectangle(roi_color, (nx, ny), (nx + nw, ny + nh), (255, 0, 0), 3)
		roi_nose = roi_gray[ny: ny+nh, nx:nx+nw]
		mustache2 = cv2.resize(mustache.copy(),(nw,int(0.5*ny)))

		mw, mh, mc = mustache2.shape
		for i in range(0,mw):
			for j in range(0,mh):
				if mustache2[i,j][3] != 0:
					roi_color[ny + int(nh/2) + i, nx+j] = mustache2[i,j]

 

#Display resulting frame
img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

cv2.imshow('Image',img)



img = np.reshape(img , (-1,3))
print((img))


cv2.waitKey(0)



cv2.destroyAllWindows()
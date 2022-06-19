
import cv2
import numpy as np


face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eyes_cascade = cv2.CascadeClassifier("./Train/third-party/haarcascade_mcs_eyepair_big.xml")
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
	
	for(ex,ey,ew,eh) in eyes:
		#cv2.rectangle(roi_color,(ex, ey), (ex+ew, ey+eh),(0,255,0),3)
		
		roi_eyes = roi_gray[ey:ey+eh, ex:ex+w]

		glasses2 = cv2.resize(glasses.copy(),(int(1.1*ew),int(2.5*eh)))
		print(glasses2.shape)
		gw, gh, gc = glasses2.shape
		for i in range(0,gw):
			for j in range(0,gh):
				if glasses2[i,j][3] != 0:
					roi_color[ ey - int(eh/1.5)+i, int(ex)+j] = glasses2[i, j]
	
	




	nose = nose_cascade.detectMultiScale(roi_gray, 1.3, 5)
	for (nx, ny, nw, nh) in nose:
		#cv2.rectangle(roi_color, (nx, ny), (nx + nw, ny + nh), (255, 0, 0), 3)
		roi_nose = roi_gray[ny: ny+nh, nx:nx+nw]
		mustache2 = cv2.resize(mustache.copy(),(nw,int(0.5*ny)))

		mw, mh, mc = mustache2.shape
		for i in range(0,mw):
			for j in range(0,mh):
				if mustache2[i,j][3] != 0:
					roi_color[ny + int(nh/1.5) + i, nx+j] = mustache2[i,j]

 

#Display resulting frame
img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
cv2.imshow('Image',img)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)





img = np.reshape(img , (-1,3))
print((img))


cv2.waitKey(0)



cv2.destroyAllWindows()
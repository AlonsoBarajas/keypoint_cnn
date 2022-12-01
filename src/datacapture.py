import cv2 as cv

cap = cv.VideoCapture(1)
framenum = 0
photonum = 1
modnum = 6
modvar = 10

while cv.waitKey(1) != 27 and photonum <= 15:
  ret,frame = cap.read()
  if ret:
    cv.imshow('Video Capture', frame)
    if framenum % 30 == 0:
      cv.imwrite('./data/'+'0'*modnum+f'{photonum}.jpg', frame)
      photonum += 1
      if photonum % modvar == 0:
        modnum -= 1
        modvar *= 10
    framenum += 1
  
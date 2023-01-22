import cv2
import numpy as np
import os 

if not os.path.exists("data"):
    os.makedirs("data")
    os.makedirs("data/train")
    os.makedirs("data/test")
    os.makedirs("data/train/0")
    os.makedirs("data/train/1")
    os.makedirs("data/train/2")
    os.makedirs("data/train/3")
    os.makedirs("data/train/4")
    os.makedirs("data/train/5")
    os.makedirs("data/train/6")
    os.makedirs("data/train/7")
    os.makedirs("data/train/8")
    os.makedirs("data/train/9")
    os.makedirs("data/train/a")
    os.makedirs("data/train/b")
    os.makedirs("data/train/c")
    os.makedirs("data/train/d")
    os.makedirs("data/train/e")
    os.makedirs("data/train/f")
    os.makedirs("data/train/g")
    os.makedirs("data/train/h")
    os.makedirs("data/train/i")
    os.makedirs("data/train/j")
    os.makedirs("data/train/k")
    os.makedirs("data/train/l")
    os.makedirs("data/train/m")
    os.makedirs("data/train/n")
    os.makedirs("data/train/o")
    os.makedirs("data/train/p")
    os.makedirs("data/train/q")
    os.makedirs("data/train/r")
    os.makedirs("data/train/s")
    os.makedirs("data/train/t")
    os.makedirs("data/train/u")
    os.makedirs("data/train/v")
    os.makedirs("data/train/w")
    os.makedirs("data/train/x")
    os.makedirs("data/train/y")
    os.makedirs("data/train/z")
    os.makedirs("data/test/0")
    os.makedirs("data/test/1")
    os.makedirs("data/test/2")
    os.makedirs("data/test/3")
    os.makedirs("data/test/4")
    os.makedirs("data/test/5")
    os.makedirs("data/test/6")
    os.makedirs("data/test/7")
    os.makedirs("data/test/8")
    os.makedirs("data/test/9")
    os.makedirs("data/test/a")
    os.makedirs("data/test/b")
    os.makedirs("data/test/c")
    os.makedirs("data/test/d")
    os.makedirs("data/test/e")
    os.makedirs("data/test/f")
    os.makedirs("data/test/g")
    os.makedirs("data/test/h")
    os.makedirs("data/test/i")
    os.makedirs("data/test/j")
    os.makedirs("data/test/k")
    os.makedirs("data/test/l")
    os.makedirs("data/test/m")
    os.makedirs("data/test/n")
    os.makedirs("data/test/o")
    os.makedirs("data/test/p")
    os.makedirs("data/test/q")
    os.makedirs("data/test/r")
    os.makedirs("data/test/s")
    os.makedirs("data/test/t")
    os.makedirs("data/test/u")
    os.makedirs("data/test/v")
    os.makedirs("data/test/w")
    os.makedirs("data/test/x")
    os.makedirs("data/test/y")
    os.makedirs("data/test/z")
    
    # Train or test 
mode = 'TRAIN'
directory = 'data/'+mode+'/'

cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    # Simulating mirror image
    frame = cv2.flip(frame, 1)
    
    # Getting count of existing images
    count = {'zero': len(os.listdir(directory+"/0")),
             'one': len(os.listdir(directory+"/1")),
             'two': len(os.listdir(directory+"/2")),
             'three': len(os.listdir(directory+"/3")),
             'four': len(os.listdir(directory+"/4")),
             'five': len(os.listdir(directory+"/5")),
             'six': len(os.listdir(directory+"/6")),
             'seven': len(os.listdir(directory+"/7")),
             'eight': len(os.listdir(directory+"/8")),
             'nine': len(os.listdir(directory+"/9")),
             'A': len(os.listdir(directory+"/a")),
             'B': len(os.listdir(directory+"/b")),
             'C': len(os.listdir(directory+"/c")),
             'D': len(os.listdir(directory+"/d")),
             'E': len(os.listdir(directory+"/e")),
             'F': len(os.listdir(directory+"/f")),
             'G': len(os.listdir(directory+"/g")),
             'H': len(os.listdir(directory+"/h")),
             'I': len(os.listdir(directory+"/i")),
             'J': len(os.listdir(directory+"/j")),
             'K': len(os.listdir(directory+"/k")),
             'L': len(os.listdir(directory+"/l")),
             'M': len(os.listdir(directory+"/m")),
             'N': len(os.listdir(directory+"/n")),
             'O': len(os.listdir(directory+"/o")),
             'P': len(os.listdir(directory+"/p")),
             'Q': len(os.listdir(directory+"/q")),
             'R': len(os.listdir(directory+"/r")),
             'S': len(os.listdir(directory+"/s")),
             'T': len(os.listdir(directory+"/t")),
             'U': len(os.listdir(directory+"/u")),
             'V': len(os.listdir(directory+"/v")),
             'W': len(os.listdir(directory+"/w")),
             'X': len(os.listdir(directory+"/x")),
             'Y': len(os.listdir(directory+"/y")),
             'Z': len(os.listdir(directory+"/z"))}
             
    
    # Printing the count in each set to the screen
    height=960
    width=1000
    blank_image = np.zeros((height,width,3), np.uint8)  
    blank_image[:]=(0,0,0)
    cv2.putText( blank_image, "MODE : "+mode, (500, 30),cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    cv2.putText( blank_image, "IMAGE COUNT", (8, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    cv2.putText( blank_image, "ZERO : "+str(count['zero']), (8, 130), cv2.FONT_HERSHEY_SIMPLEX,1, (255,255,255), 2)
    cv2.putText( blank_image, "ONE : "+str(count['one']), (8, 160), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    cv2.putText( blank_image, "TWO : "+str(count['two']), (8, 190), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    cv2.putText( blank_image, "THREE : "+str(count['three']), (8, 220), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    cv2.putText( blank_image, "FOUR : "+str(count['four']), (8, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    cv2.putText( blank_image, "FIVE : "+str(count['five']), (8, 280), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    cv2.putText( blank_image, "SIX : "+str(count['six']), (8, 310), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    cv2.putText( blank_image, "SEVEN : "+str(count['seven']), (8, 310), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    cv2.putText( blank_image, "EIGHT : "+str(count['eight']), (8, 310), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    cv2.putText( blank_image, "NINE : "+str(count['nine']), (8, 310), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    cv2.putText( blank_image, "alpha- A : "+str(count['A']), (8, 340), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    cv2.putText( blank_image, "alpha- B : "+str(count['B']), (8, 370), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    cv2.putText( blank_image, "alpha- C : "+str(count['C']), (8, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    cv2.putText( blank_image, "alpha- D : "+str(count['D']), (8, 430), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    cv2.putText( blank_image, "alpha- E : "+str(count['E']), (8, 460), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    cv2.putText( blank_image, "alpha- F : "+str(count['F']), (8, 490), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    cv2.putText( blank_image, "alpha- G : "+str(count['G']), (8, 510), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    cv2.putText( blank_image, "alpha- H : "+str(count['H']), (8, 540), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    cv2.putText( blank_image, "alpha- I : "+str(count['I']), (8, 570), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    cv2.putText( blank_image, "alpha- J : "+str(count['J']), (8, 600), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    cv2.putText( blank_image, "alpha- K : "+str(count['K']), (8, 630), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    cv2.putText( blank_image, "alpha- L : "+str(count['L']), (8, 660), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    cv2.putText( blank_image, "alpha- M : "+str(count['M']), (8, 690), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    cv2.putText( blank_image, "alpha- N : "+str(count['N']), (8, 720), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    cv2.putText( blank_image, "alpha- O : "+str(count['O']), (8, 750), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    cv2.putText( blank_image, "alpha- P : "+str(count['P']), (8, 780), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    cv2.putText( blank_image, "alpha- Q : "+str(count['Q']), (8, 810), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    cv2.putText( blank_image, "alpha- R : "+str(count['R']), (8, 840), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    cv2.putText( blank_image, "alpha- S : "+str(count['S']), (8, 870), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    cv2.putText( blank_image, "alpha- T : "+str(count['T']), (8, 900), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    cv2.putText( blank_image, "alpha- U : "+str(count['U']), (8, 930), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    cv2.putText( blank_image, "alpha- V : "+str(count['V']), (8, 960), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    cv2.putText( blank_image, "alpha- W : "+str(count['W']), (8, 990), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    cv2.putText( blank_image, "alpha- X : "+str(count['X']), (8, 1020), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    cv2.putText( blank_image, "alpha- Y : "+str(count['Y']), (8, 1050), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    cv2.putText( blank_image, "alpha- Z : "+str(count['Z']), (8, 1080), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    cv2.imshow('3 Channel Window', blank_image)   
    

    # Coordinates of the ROI
    x1 = int(0.5*frame.shape[1])
    y1 = 10
    x2 = frame.shape[1]-10
    y2 = int(0.5*frame.shape[1])
    # Drawing the ROI
    # The increment/decrement by 1 is to compensate for the bounding box
    cv2.rectangle(frame, (x1-1, y1-1), (x2+1, y2+1), (255,0,0) ,1)
    # Extracting the ROI
    roi = frame[y1:y2, x1:x2]
    roi = cv2.resize(roi, (512,512)) 
 
    cv2.imshow("Frame", frame)
    
    #_, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
    #kernel = np.ones((1, 1), np.uint8)
    #img = cv2.dilate(mask, kernel, iterations=1)
    #img = cv2.erode(mask, kernel, iterations=1)
    # do the processing after capturing the image!
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, roi = cv2.threshold(roi, 120, 255, cv2.THRESH_BINARY)
    cv2.imshow("ROI", roi)
    
    interrupt = cv2.waitKey(10)
    if interrupt & 0xFF == 27: # esc key
        break
    if interrupt & 0xFF == ord('0'):
        cv2.imwrite(directory+'0/'+str(count['zero'])+'.jpg', roi)
    if interrupt & 0xFF == ord('1'):
        cv2.imwrite(directory+'1/'+str(count['one'])+'.jpg', roi)
    if interrupt & 0xFF == ord('2'):
        cv2.imwrite(directory+'2/'+str(count['two'])+'.jpg', roi)
    if interrupt & 0xFF == ord('3'):
        cv2.imwrite(directory+'3/'+str(count['three'])+'.jpg', roi)
    if interrupt & 0xFF == ord('4'):
        cv2.imwrite(directory+'4/'+str(count['four'])+'.jpg', roi)
    if interrupt & 0xFF == ord('5'):
        cv2.imwrite(directory+'5/'+str(count['five'])+'.jpg', roi)
    if interrupt & 0xFF == ord('6'):
        cv2.imwrite(directory+'6/'+str(count['six'])+'.jpg', roi)
    if interrupt & 0xFF == ord('7'):
        cv2.imwrite(directory+'7/'+str(count['seven'])+'.jpg', roi)
    if interrupt & 0xFF == ord('8'):
        cv2.imwrite(directory+'8/'+str(count['eight'])+'.jpg', roi)
    if interrupt & 0xFF == ord('9'):
        cv2.imwrite(directory+'9/'+str(count['nine'])+'.jpg', roi)
    if interrupt & 0xFF == ord('a'):
        cv2.imwrite(directory+'a/'+str(count['A'])+'.jpg', roi)
    if interrupt & 0xFF == ord('b'):
        cv2.imwrite(directory+'b/'+str(count['B'])+'.jpg', roi)
    if interrupt & 0xFF == ord('c'):
        cv2.imwrite(directory+'c/'+str(count['C'])+'.jpg', roi)
    if interrupt & 0xFF == ord('d'):
        cv2.imwrite(directory+'d/'+str(count['D'])+'.jpg', roi)
    if interrupt & 0xFF == ord('e'):
        cv2.imwrite(directory+'e/'+str(count['E'])+'.jpg', roi)
    if interrupt & 0xFF == ord('b'):
        cv2.imwrite(directory+'f/'+str(count['F'])+'.jpg', roi)
    if interrupt & 0xFF == ord('g'):
        cv2.imwrite(directory+'g/'+str(count['G'])+'.jpg', roi)
    if interrupt & 0xFF == ord('h'):
        cv2.imwrite(directory+'h/'+str(count['H'])+'.jpg', roi)
    if interrupt & 0xFF == ord('i'):
        cv2.imwrite(directory+'i/'+str(count['I'])+'.jpg', roi)
    if interrupt & 0xFF == ord('j'):
        cv2.imwrite(directory+'j/'+str(count['J'])+'.jpg', roi)
    if interrupt & 0xFF == ord('k'):
        cv2.imwrite(directory+'k/'+str(count['K'])+'.jpg', roi)
    if interrupt & 0xFF == ord('l'):
        cv2.imwrite(directory+'l/'+str(count['L'])+'.jpg', roi)
    if interrupt & 0xFF == ord('m'):
        cv2.imwrite(directory+'m/'+str(count['M'])+'.jpg', roi)
    if interrupt & 0xFF == ord('n'):
        cv2.imwrite(directory+'n/'+str(count['N'])+'.jpg', roi)
    if interrupt & 0xFF == ord('o'):
        cv2.imwrite(directory+'o/'+str(count['O'])+'.jpg', roi)
    if interrupt & 0xFF == ord('p'):
        cv2.imwrite(directory+'p/'+str(count['P'])+'.jpg', roi)
    if interrupt & 0xFF == ord('q'):
        cv2.imwrite(directory+'q/'+str(count['Q'])+'.jpg', roi)
    if interrupt & 0xFF == ord('r'):
        cv2.imwrite(directory+'r/'+str(count['R'])+'.jpg', roi)
    if interrupt & 0xFF == ord('s'):
        cv2.imwrite(directory+'s/'+str(count['S'])+'.jpg', roi)
    if interrupt & 0xFF == ord('t'):
        cv2.imwrite(directory+'t/'+str(count['T'])+'.jpg', roi)
    if interrupt & 0xFF == ord('u'):
        cv2.imwrite(directory+'u/'+str(count['U'])+'.jpg', roi)
    if interrupt & 0xFF == ord('v'):
        cv2.imwrite(directory+'v/'+str(count['V'])+'.jpg', roi)
    if interrupt & 0xFF == ord('w'):
        cv2.imwrite(directory+'w/'+str(count['W'])+'.jpg', roi)
    if interrupt & 0xFF == ord('x'):
        cv2.imwrite(directory+'x/'+str(count['X'])+'.jpg', roi)
    if interrupt & 0xFF == ord('y'):
        cv2.imwrite(directory+'y/'+str(count['Y'])+'.jpg', roi)
    if interrupt & 0xFF == ord('z'):
        cv2.imwrite(directory+'z/'+str(count['Z'])+'.jpg', roi)

cap.release()
cv2.destroyAllWindows()
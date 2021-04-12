# imports
import cv2
import numpy as np
import time

from bird_eye_view import bird_eye_view # import bird eyeview function




# imports for mailing function
import email, smtplib
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
import email_template as temp

#-------------------- mailing function to alert ------------------------------------------------------------------
def email(f_path,receiver_email,c):
    subject = "Social Distance Alert"
    body = "Social distance alert"
    sender_email = "distacingbyML@gmail.com"
    # receiver_email = "example@gmail.com"
    password = "distacingbyML2021"

    # Create a multipart message and set headers
    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = receiver_email
    message["Subject"] = subject
    message["Bcc"] = receiver_email  # Recommended for mass emails
    html=temp.get(c)
    # html.format(c1="10",c2='20',c3='90')
    part2 = MIMEText(html, 'html')
    message.attach(part2)
    fp = open(f_path, 'rb')
    msgImage = MIMEImage(fp.read())
    fp.close()

    # Define the image's ID as referenced above
    msgImage.add_header('Content-ID', '<image1>')
    message.attach(msgImage)
    server = smtplib.SMTP('smtp.gmail.com',587)
    server.starttls()
    server.login(sender_email, password)
    text = message.as_string()
    server.sendmail(sender_email, receiver_email, text)
    server.quit()
# region of interest points 
ROI_points = []
# -------------------------------- calculate the distance between each person--------------------------------------------------
def calcualte_distnace(person1,person2,v_scale_distance,h_scale_distance):

          h = abs(person1[1]-person2[1])
          w = abs(person1[0]-person2[0])
          
          new_dis_v = float((w/v_scale_distance)*180)
          new_dis_h = float((h/h_scale_distance)*180)

          distance=int(np.sqrt(((new_dis_h)**2) + ((new_dis_v**2))))
          return distance
#----------------- find the risk condition of each person---------------------------------------------------------------------
def find_status(distance):
  status=0

  if distance>180:
    status=2
  elif distance>150:
    status=1

  return status
#------------------------------ calculate distance between all pair of people---------------------------------------------------
def get_all_distances(boxes, center_points, dist_v, dist_h):

  dist_mat=[]
  all_pairs=[]
  
  for i in range(len(center_points)):

    for j in range(len(center_points)):

      if i!=j:

          dist=calcualte_distnace(center_points[i],center_points[j],dist_v, dist_h)

          status=find_status(dist)

          dist_mat.append([center_points[i],center_points[j],status])
          all_pairs.append([boxes[i],boxes[j],status])

  return dist_mat,all_pairs



#--------------Function gives count for humans at high risk, low risk and no risk--------------------------------------------
# input: 
# distance_matrix:list of [box1,box2,status]. status =0 then high risk , 1 then low risk , 2 then No risk
# output:
# list[no of person in high,low, No risk]

def is_new_item(r,y,g,person):
  if (person not in r) and (person not in g) and (person not in y):
      return True

def get_safe_people(distance_matrix,r,y,g):
  green=0

  for i in range(len(distance_matrix)):
    if distance_matrix[i][2] == 2:
            if is_new_item(r,y,g,distance_matrix[i][0]):
                g.append(distance_matrix[i][0])
                green+=1
            if is_new_item(r,y,g,distance_matrix[i][1]):
                g.append(distance_matrix[i][1])
                green+=1

  return g,green

def get_low_risk_people(distance_matrix,r,y,g):
  yellow=0

  for i in range(len(distance_matrix)):

    if distance_matrix[i][2] == 1:
            if is_new_item(r,y,g,distance_matrix[i][0]):
                y.append(distance_matrix[i][0])
                yellow+=1
            if is_new_item(r,y,g,distance_matrix[i][1]):
                y.append(distance_matrix[i][1])
                yellow+=1

  return y,yellow

def get_high_risk_people(distance_matrix,r,y,g):
  red=0
  for i in range(len(distance_matrix)):

    if distance_matrix[i][2] == 0:
            if is_new_item(r,y,g,distance_matrix[i][0]):
                r.append(distance_matrix[i][0])
                red+=1
            if is_new_item(r,y,g,distance_matrix[i][1]):
                r.append(distance_matrix[i][1])
                red+=1

  return r,red

    
def risk_counter(distance_matrix):
    
    r = []
    g = []
    y = []
    
    red=0
    green=0
    yellow=0

    r,red=get_high_risk_people(distance_matrix,r,y,g)

    y,yellow=get_low_risk_people(distance_matrix,r,y,g)
    
    g,green=get_safe_people(distance_matrix,r,y,g)
   
    return (red,yellow,green)

# ----------------------------------- draw bounding boxes based on risk factor for person-------------------------------------
#input 
# frame: image
# dis_matrix: list of [box1,box2,status]. status =0 then high risk , 1 then low risk , 2 then No risk
# boxes : bounding box. # boxes : list of box[x,y,width,height]
# risk_count= number of person in high,low,no Risk

#output:
#output image
#draw bounding boxes based on risk factor for person in a frame and draw visualising line lines between


def camera_view(frame, dis_matrix, boxes, risk_count):

  
  for box in boxes:
    (x, y, w, h) = box[0:4]
    cv2.rectangle(frame,(x,y),(x+w,y+h),(0, 255, 0),2)
    
  for i in dis_matrix:
    box = i[0:2]
    if i[2] == 1: #yellow 
      x, y, w, h = box[0][0], box[0][1], box[0][2], box[0][3]
      cv2.rectangle(frame,(x,y),(x+w,y+h),(0, 255, 255),2)
      x, y, w, h = box[1][0], box[1][1], box[1][2], box[1][3]
      cv2.rectangle(frame,(x,y),(x+w,y+h),(0, 255, 255),2)

  for i in dis_matrix:
    box = i[0:2]
    if i[2] == 0: #red
      x, y, w, h = box[0][0], box[0][1], box[0][2], box[0][3]
      cv2.rectangle(frame,(x,y),(x+w,y+h),(0, 0, 255),2)
      x, y, w, h = box[1][0], box[1][1], box[1][2], box[1][3]
      cv2.rectangle(frame,(x,y),(x+w,y+h),(0, 0, 255),2)
  return frame
#---------------------------------------person dection by YOLOv3 model------------------------------------------------
def Yolov3(frame,ln1,net):
        hight,width,_ = frame.shape
        blob = cv2.dnn.blobFromImage(frame, 1/255,(416,416),(0,0,0),swapRB = True,crop= False)

        net.setInput(blob)

        layerOutputs = net.forward(ln1)

        boxes =[]
        confidences = []
        class_ids = []

        for output in layerOutputs:
            for detection in output:
                score = detection[5:]
                class_id = np.argmax(score)
                confidence = score[class_id]
                if class_id==0:
                    if confidence > 0.5:
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * hight)
                        w = int(detection[2] * width)
                        h = int(detection[3]* hight)
                        x = int(center_x - w/2)
                        y = int(center_y - h/2)
                        boxes.append([x,y,w,h])
                        confidences.append((float(confidence)))
                        class_ids.append(class_id)


        indexes = cv2.dnn.NMSBoxes(boxes,confidences,.5,.5)
        font = cv2.FONT_HERSHEY_PLAIN

        boxes1 = []
        val1=set()
        val2=set()
        if len(indexes) > 0:
            for i in indexes.flatten():
                if boxes[i][0] not in val1 and boxes[i][1] not in val2:
                    val1.add(boxes[i][0])
                    val2.add(boxes[i][1])
                    boxes1.append(boxes[i])

        return boxes1   
       
            
#-------------------------- Mouse click Event ----------------------------------------------------------

def calibration_with_mouse(event, x, y, flags, param):

    global ROI_points,image
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(ROI_points) < 4:
            cv2.circle(image, (x, y), 5, (0, 0, 255), 5)
        else:
            cv2.circle(image, (x, y), 5, (0, 255, 0), 5)
            
        if len(ROI_points) >= 1 and len(ROI_points) <= 3:
            cv2.line(image, (x, y), (ROI_points[len(ROI_points)-1][0], ROI_points[len(ROI_points)-1][1]), (255, 0, 0), 1)
            if len(ROI_points) == 3:
                cv2.line(image, (x, y), (ROI_points[0][0], ROI_points[0][1]), (255, 0, 0), 1)
        ROI_points.append((x, y))

#------------------------------perpective transformation-----------------------------------------------
def find_transformed_points(boxes, prespective_transform):
    
        bottom_points = []
        for box in boxes:
            pnts = np.array([[[int(box[0]+(box[2]*0.5)),int(box[1]+box[3])]]] , dtype="float32")
            bd_pnt = cv2.perspectiveTransform(pnts, prespective_transform)[0][0]
            pnt = [int(bd_pnt[0]), int(bd_pnt[1])]
            bottom_points.append(pnt)
            
        return bottom_points
#---------------------------------- dectect the person with yolov3 function in each frame----------------------------------
def detection(vid_path,mail_id,thres,net,ln1):

    global image

    def calling_mouse_click_event(frame):
         while True:
            global image
            image = frame
            cv2.imshow("image", image)
            cv2.waitKey(1)
            if len(ROI_points) == 8:
                cv2.destroyWindow("image")
                break

    precount,count = True,0
    video = cv2.VideoCapture(vid_path)    

    # collecting the required values form video properties

    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = int(video.get(cv2.CAP_PROP_FPS))

    points = []
    scale_w, scale_h = float(400/width), float(400/height)

    while True:

        (grabbed, frame) = video.read()

        if not grabbed:
            print('completed the task or error in loading video')
            break
        if count%5!=0:
            count+=1
            continue
        if count == 0:
            calling_mouse_click_event(frame)
            points = ROI_points
            print(points)
            print(frame.shape[:]) 
            
        init_pts = np.float32(np.array(points[:4]))
        frame_height = frame.shape[0]
        frame_width = frame.shape[1]
        final_pts = np.float32([[0,frame_height], [frame_width, frame_height], [frame_width, 0], [0, 0]])
      

        prespective_transform_matrix = cv2.getPerspectiveTransform(init_pts, final_pts)
      
        pts = np.float32(np.array([points[4:7]]))
        transformed_pts= cv2.perspectiveTransform(pts, prespective_transform_matrix)[0]
        
        distance_w = np.sqrt((transformed_pts[0][0] - transformed_pts[1][0]) ** 2 + (transformed_pts[0][1] - transformed_pts[1][1]) ** 2)
        distance_h = np.sqrt((transformed_pts[0][0] - transformed_pts[2][0]) ** 2 + (transformed_pts[0][1] - transformed_pts[2][1]) ** 2)
      
        ROI = np.array(points[:4], np.int32)
        cv2.polylines(frame, [ROI], True, (70, 70, 70), thickness=2)

        # YOLO v3
        boxes1=Yolov3(frame,ln1,net)

        if len(boxes1) == 0:

            count = count + 1
            continue
        
        person_points = find_transformed_points(boxes1, prespective_transform_matrix)
        
        distances_mat, bxs_mat =get_all_distances(boxes1, person_points, distance_w, distance_h)
        risk_count = risk_counter(distances_mat)

        frame1 = np.copy(frame)
        
        bird_image = bird_eye_view(frame, distances_mat, person_points, scale_w, scale_h, risk_count)
        img = camera_view(frame1, bxs_mat, boxes1, risk_count)

        #img = camera_view1(frame1, boxes1)
        if count != 0:
            # output_movie.write(img)
            # bird_movie.write(bird_image)

            cv2.imshow('Bird Eye View', bird_image)
            cv2.imshow('social distancing view', img)

            cv2.imwrite("output/frame.jpg", img)
            cv2.imwrite("output/bird_eye_view.jpg", bird_image)
            percent=risk_count[0]/sum(risk_count)*100
            # print(percent)
            if percent>thres and precount:
                  precount=False
                  email("output/frame.jpg",mail_id,risk_count)

            # fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            # writer = cv2.VideoWriter("output.avi", fourcc, 25,(bird_image.shape[1], bird_image.shape[0]), True)

	# if the video writer is not None, write the frame to the output video file
            # if writer is not None:
            #   writer.write(bird_image)

        count = count + 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
      
    video.release()
    cv2.destroyAllWindows() 

#--------------------------- main funcion for testion purpose-------------------------------------------------
def main():
    weightsPath ="models/yolov3.weights"
    configPath = "models/yolov3.cfg"

    

    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
    ln = net.getLayerNames()
    ln1 = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    cv2.namedWindow("image")
    cv2.setMouseCallback("image", calibration_with_mouse) 

    detection("ex.mp4", net, ln1)

if __name__== "__main__":
    main()
# ----------------- main function accessed by ui fuction---------------------------------------------------
def exmain(path,mail_id,thres):
    weightsPath ="models/yolov3.weights"
    configPath = "models/yolov3.cfg"

    

    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
    ln = net.getLayerNames()
    ln1 = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    cv2.namedWindow("image")
    cv2.setMouseCallback("image", calibration_with_mouse) 
    if path=="0":
      path=0
    detection(path,mail_id,thres, net, ln1)
#------------------------------------------------------------------------------------------------------------------
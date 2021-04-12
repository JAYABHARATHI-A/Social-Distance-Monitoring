# Social-Distance-Monitoring
Machine Learning techniques to monitor real-time social distancing at public places especially when there are many social interactions being taken place. 

  The ongoing Covid-19 pandemic is a global disaster that disrupted the normal life of the people and caused more than 2 million deaths worldwide. Among various precautionary measures, Social Distancing is proved to be an effective measure in contracting the infection spread in the society. Now, as many cities in our country are moving back to normal cautiously, people have been instructed to follow social distancing rules as they venture out. It is important to monitor the social distance and wearing masks at public places and take actions accordingly. If most people follow them, then more places can be opened safely. However, if there are many violations then it may be safer to close. This is exactly what happened in Andhra Pradesh where schools and colleges were reopened after seven months in November but were closed within the week since too many people were flouting rules related to wearing masks and socially distancing. The state Government detected this by using officers to monitor. But manual monitoring may not always be an efficient and effective solution. So, we employ AI and machine learning techniques to monitor if people are following social distancing guidelines. Most of the cities already have cameras installed at public places which can be used to monitor and facilitate social distancing. The monitoring system analyses these footages, to monitor the social distance between the individuals in real time and can act based on the analysis performed on these footages. The aim of the paper is to build a social distancing tool to avoid transmission of contagious diseases through Computer Vision, Pattern Recognition using Machine Learning. 
  
The proposed tool has following features:
      Identify humans in the frame with yolov3 (You Only Look Once).
      Calculates the distance among individuals who are identified in the frame.
      Shows the number of people who are at ‘High risk’, ‘Low risk’ and ‘Not at risk’

**Download the Yolo weights (https://pjreddie.com/media/files/yolov3.weights) and move it to SocialDistanceMonitoring/Model directory**


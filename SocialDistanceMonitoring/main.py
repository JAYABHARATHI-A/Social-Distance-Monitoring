# front end UI
from tkinter import *
from functools import partial
from yolo_dection import *


 # UI function
def ui():
  win = Tk()  # create the window
  win.title("Social Distance Monitoring") # set the tilte for created window
  win.geometry("430x300")  # set the dimension of window

  def printer(s1,s2,s3): # call the dection and monitoring function
       win.destroy()# destroy the window
      #  print(s1.get(),s2.get(),s3.get())
       exmain(s1.get(),s2.get(),int(s3.get())) # call the detection function
    
  name = Label(win, text = "Video path or URL").place(x = 40,y = 50)  # entry for video path
    
  email = Label(win, text = "Email id ").place(x = 40, y = 90)  # entry for email id to sent alert
    
  thres = Label(win, text = "Minimum Thresould").place(x = 40, y = 130)   # entry for minimum  thresould in percentage
  # text varialbe for each entry
  s1=StringVar(win, value='input/ex.mp4')
  s2=StringVar(win, value='bharathiaj7071@gmail.com')
  s3=StringVar(win, value=45)
  # fix position the entrys in window
  entry1 = Entry(win, textvariable=s1).place(x = 180, y = 50) 
  entry2 = Entry(win, textvariable=s2).place(x = 180, y = 90)  
  entry3 = Entry(win, textvariable=s3).place(x = 180, y = 130)  

  printer = partial(printer, s1,s2,s3)   # argument passing to function 
  # run button
  submitbtn = Button(win, text = "RUN", activeforeground = "blue",command=printer).place(x = 100, y = 170) 

    
  win.mainloop() # loop the window 
if __name__ == "__main__":
  ui() # call ui function

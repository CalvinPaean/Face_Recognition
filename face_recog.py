# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 23:24:38 2017

@author: biank
https://medium.com/@ageitgey/machine-learning-is-fun-part-4-modern-face-recognition-with-deep-learning-c3cffc121d78
"""
import sys,os,dlib,glob,numpy
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from tkinter import *
from skimage import io


def run_program():
    # 1.Human face key points detector
    predictor_path = e1.get()
    
    # 2.Human face recognition model(coefficients)
    face_rec_model_path = e2.get()
    
    # 3.Folder of candidates' faces
    faces_folder_path = e3.get()
    
    # 4.Faces needed to be recognized
    img_path = e4.get()
    
    #show the image you want to test
    img = mpimg.imread(img_path)
    implot = plt.imshow(img)
    plt.show()
    
    # 1. return a detector that finds human faces that are looking more or less towards the camera.
    detector = dlib.get_frontal_face_detector()
    
    # 2.return a predictor that takes in an image region containing some object and outputs a set 
    #of point locations that define the pose of the object.
    sp = dlib.shape_predictor(predictor_path)
    
    # 3. Loading face recognition model
    """
     maps human faces into 128D vectors where pictures of the same person are mapped near 
     to each other and pictures of different people are mapped far apart. 
    """
    facerec = dlib.face_recognition_model_v1(face_rec_model_path)
    
    # list of candidates' faces
    descriptors = []
    
    #The glob module finds all the pathnames matching a specified pattern
    for f in glob.glob(os.path.join(faces_folder_path, "*.jpg")):
        print("Processing file: {}".format(f))
        img = io.imread(f)
      
        # 1.face detection
        """
        The 1 in the second argument indicates that we should upsample the image
        1 time. This will make everything bigger and allow us to detect more
        faces
        """
        #dets:the positions of objects in an image
        dets = detector(img, 1)
        #print("Number of faces detected: {}".format(len(dets)))
        
        for k, d in enumerate(dets): 
            
            # 2.key points detection
            shape = sp(img, d)
            
            # 3.describe the sub-retrival of image ==> 128D vector
            face_descriptor = facerec.compute_face_descriptor(img, shape)
    
            # convert to numpy array
            v = numpy.array(face_descriptor)  
            descriptors.append(v)
    
    # process faces needed to be recognized
    img = io.imread(img_path)
    dets = detector(img, 1)
    
    dist = []
    for k, d in enumerate(dets):
        shape = sp(img, d)
        face_descriptor = facerec.compute_face_descriptor(img, shape)
        d_test = numpy.array(face_descriptor)
    
        # compute the Euclidean distance
        for i in descriptors:
            dist_ = numpy.linalg.norm(i-d_test)
            dist.append(dist_)
    
    # candidates
    candidate = []
    for f in glob.glob(os.path.join(faces_folder_path, "*.jpg")):
        candidate.append(f[18:-4])
    # form a dict from candidates and distances
    c_d = dict(zip(candidate,dist))
    cd_sorted = sorted(c_d.items(), key=lambda d:d[1])
    
    result_path = './candidate-faces/'+cd_sorted[0][0] + '.jpg'
    img2 = mpimg.imread(result_path)
    implot = plt.imshow(img2)
    plt.show()
    
    print ("\n That person is: ",cd_sorted[0][0]) 
    dlib.hit_enter_to_continue()
    
#Construct the GUI component for input paths
master = Tk()
master.title('TTP Final Project')
#four labels
#label 1
Label(master, text = "Loc of Face Key Points Data:").grid(row=0)
#label 2
Label(master, text = "Loc of Face Recgntion Data:").grid(row=1)
#label 3
Label(master, text = "Loc of Candidates:").grid(row=2)
#label 4
Label(master, text = "Loc of Test Image:").grid(row = 3)

#four text areas
#text area1
e1 = Entry(master)
#text area2
e2 = Entry(master)
#text area3
e3 = Entry(master)
#text area4
e4 = Entry(master)

#set each text areas initial value
e1.insert(10,"1.dat")
e2.insert(10,"2.dat")
e3.insert(20,"./candidate-faces")
e4.insert(10,"a1.jpg")

#set the locations of text areas
e1.grid(row=0,column=1)
e2.grid(row=1,column=1)
e3.grid(row=2,column=1)
e4.grid(row=3,column=1)

#button quit
Button(master, text = 'Quit', command=master.destroy).grid(row=4,column=0,sticky=W,pady=4)
#button predict
Button(master, text = 'Predict', command=run_program).grid(row=4, column=1,sticky=W,pady=4)
#run the GUI
mainloop()
    
#1.dat 2.dat ./candidate-faces a1.jpg
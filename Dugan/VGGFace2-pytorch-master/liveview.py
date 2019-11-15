import datetime
import math
import os
import gc
import time

import cv2

import numpy as np
import torch
from torch.autograd import Variable

from correlationmodel import CorrelationModel

import utils
import tqdm

class LiveView(object):

    def __init__(self, cuda, model, log_file, feature_dir, flatten_feature=True, print_freq=1):
      """
      :param cuda:
      :param model:
      :param log_file: log file name. logs are appended to this file.
      :param feature_dir:
      :param flatten_feature:
      :param print_freq:
      """
      self.cuda = cuda
      self.log_file = log_file
      self.feature_dir = feature_dir
      self.flatten_feature = flatten_feature
      self.print_freq = print_freq

      self.timestamp_start = datetime.datetime.now()

      self.users = []

      nput = 'y'
      # Take 5 face shots in 224x224
      # Front, side:left, side:right, looking down, looking up
      while nput is 'y':
        self.users.append(self.faceCapture())
        #print("There are currently %d user(s), another? y for yes, anything else for no."%len(self.users))
        nput='n'
        #nput = input()

      self.model = CorrelationModel(model,self.users)

    def faceCapture(self):
      cam = cv2.VideoCapture(0)
      cv2.namedWindow("Current Face")
      img_counter = 0
      faces = ['forward','up','left','right','down']
      imgs = []

      while True and len(imgs) < 2:
        ret, frame = cam.read()
        if not ret:
            print("Webcam failure.")
            break
        k = cv2.waitKey(1)

        # Draw rectangle in center of frame to assist user in making a good
        #   capture.
        y = frame.shape[0]//2 - 122
        x = frame.shape[1]//2 - 122
        w = 3
        tl = (x-w,y-w)
        br = (x+247 + w,y+244 + w)
        cv2.rectangle(frame,tl,br,(0,0,255),w)
        cv2.imshow("Current Face", frame)

        if k%256 == 27:
          # ESC pressed
          print("Escape hit, closing...")
          break
        elif k%256 == 32 and len(imgs) < 2:
          # SPACE pressed
          frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
          imgs.append(frame[x:x+244,y:y+244])
          print("Added face perspective.")
          img_counter += 1
        # END LOOP

      cam.release()
      cv2.destroyAllWindows()
      return imgs

    def view(self):
      cam2 = cv2.VideoCapture(0)
      cv2.namedWindow("Found Faces")
      while True:
        ret, frame = cam2.read()
        if not ret:
            print("Webcam failure.")
            break
        k = cv2.waitKey(1)

        if k%256 == 27:
          # ESC pressed
          print("Escape hit, closing...")
          break
        frame2 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.model.correlate(frame2,self.users)

        # RESULTS IS IN SHAPE [User][Perspective](CorrVal,TLPosition) = [User][Perspective](15,25)
        for user in results:
          p=0
          pos = np.zeros(len(user))
          for pers in user:

            pos[p] = torch.max(pers)
            p+=1
          perspective = np.argmax(pos)
          idx = torch.argmax(user[perspective])
          tl0 = (idx//user[perspective].size()[1])/user[perspective].size()[0]*frame2.shape[0]
          tl1 = (idx % user[perspective].size()[1])/user[perspective].size()[1]*frame2.shape[1]
          cv2.rectangle(frame2, (tl0.detach().cpu()-122,tl1.detach().cpu()-122),
                        (tl0.detach().cpu() + 122, tl1.detach().cpu() + 122),
                        (0, 0, 255), 3)
              #print(corrMax,idx)
              #tlPos = user[idx][1]
          #print(results)
      cam2.release()
      cv2.destroyAllWindows()
      '''
        
        for user in results:
          # USER SHAPE [Perspective](CorrVal,TLPos)
          corrMax = max(user)

          # If its a match...
         # if corrMax > thresh:
          idx = (user == corrMax).nonzero()
          tlPos = user[idx][1]
          cv2.rectangle(frame2,tlPos.detach().cpu(),(tlPos[0].detach().cpu()+244,tlPos[1].detach().cpu()+244),(0,0,255),3)
            # Todo: Names
          #else:
            # not found.
         #n
        # pass
      '''
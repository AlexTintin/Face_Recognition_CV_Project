#Filetoos!
import os
import cv2
import numpy as np

def faceCapture():
  cam = cv2.VideoCapture(0)
  cv2.namedWindow("Current Face")
  img_counter = 0
  faces = ['forward','up','left','right','down']
  imgs = []

  while True:
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

# This function recursivly makes directories.
def directoryFixer(directory):
  directory = "".join(str(x) for x in directory)
  try:
    os.stat(directory)
  except:
    try:
      os.mkdir(directory)
    except:
      subDir = directory.split('/')
      while (subDir[-1] == ''):
          subDir = subDir[:-1]
      newDir = ""
      for x in range(len(subDir)-1):
        newDir += (subDir[x])
        newDir += ('/')
      print("Fixing ",newDir)
      directoryFixer(newDir)
      os.mkdir(directory)

# This function finds all files of given extention in given path.
def find_files(path,ext = '.png'):
  return [path + '/' + f for f in os.listdir(path) if f.endswith(ext)]

# This function uses find_files to find ALL FILES recursivly in given path root
def parse_dir(base_directory,ext = '.png'):
  returnlist = []
  for x in os.walk(base_directory):
      x = x[0].replace('\\','/')
      # print("Walking: "+x)
      appendlist = find_files(x,ext)
      if appendlist:
        returnlist.append(appendlist)
  ret_list = []
  for r in returnlist:
    for s in r:
      ret_list.append(s)
  return ret_list

def example_opener():
  ex_paths = [f.path for f in os.scandir(folder) if f.is_dir()]
  names = [ex_path.split('/')[-1] for ex_path in ex_paths]
  examples = []

  for ex_path in ex_paths:
    p_paths = find_files(ex_path)
    perspectives = [cv2.imread(path) for path in p_paths]
    examples.append(name,perspectives)

  return examples

def main():
  exit = 'n'
  while exit == 'n':
    examples = []
    print("Adding new user.")
    name = input("Enter name for this example: ")
    examples.append((name,faceCapture()))
    exit = input("Exit? n to keep going: ")

  for example in examples:
    name, perspectives = example
    path = os.path.join('./','data',name)
    directoryFixer(path)

    for x in range(len(perspectives)):
      cv2.imwrite('%d.png'%x,perspectives[x])

if __name__== "__main__":
  main()

import numpy as np
# import matplotlib.pyplot as plt 
from math import exp
import cv2, os, time, threading

class BGSubtraction():
  def __init__(self, input_dir, big_patch=15, small_patch=2):
    self.input_dir = input_dir 
    self.big_patch = big_patch
    self.small_patch = small_patch
    self.cnt = 50
    self.gt = 300 * 2 / (1 + exp(-1*(self.cnt - 100)/20))
    self.x_axis = [1.5, 5.5, 9.5, 13.5, 17.5, 21.5, 25.5, 29.5, 33.5, 37.5, 41.5, 45.5, 49.5, 53.5, 57.5, 61.5, 65.5, 69.5, 73.5, 77.5, 81.5, 85.5, 89.5, 93.5, 97.5, 101.5, 105.5, 109.5, 113.5, 117.5, 121.5, 125.5, 129.5, 133.5, 137.5, 141.5, 145.5, 149.5, 153.5, 157.5, 161.5, 165.5, 169.5, 173.5, 177.5, 181.5, 185.5, 189.5, 193.5, 197.5, 201.5, 205.5, 209.5, 213.5, 217.5, 221.5, 225.5, 229.5, 233.5, 237.5, 241.5, 245.5, 249.5, 253.5]
    self.x_axis = np.array(self.x_axis)
    self.Nd = 64
    self.Bd = 255/self.Nd

    self.my_dict = {}

    print("Preprocessing starting...")
    for i in range(256):
      self.gaussian(i, b=self.Bd)
    print("Preprocessing ended...")

    self.doit()


  def gaussian(self, x, b=1):
    key = x 
    if self.my_dict.get(x) is not None:
      return self.my_dict.get(x) 
    
    x = self.x_axis - x 
    self.my_dict[key] = np.exp(-x**2/(2*(b/2)**2))/((b/2)*np.sqrt(2*np.pi))
    return self.my_dict[key]


  def init_bg_model(self):
         
    self.frames = os.listdir(self.input_dir)
    self.frames.sort()

    self.img = cv2.imread(f"{self.input_dir}/{self.frames[0]}", 1)
    self.height, self.width = self.img[:,:,0].shape

    self.grad1 = np.ones((self.height, self.width))
    self.grad2 = np.ones((self.height, self.width))
    self.grad3 = np.ones((self.height, self.width))

    self.prev_bg_model1 = np.zeros((self.height, self.width, self.Nd))
    self.prev_bg_model2 = np.zeros((self.height, self.width, self.Nd))
    self.prev_bg_model3 = np.zeros((self.height, self.width, self.Nd))

    num_of_starting_frames = 1
    for i in range(num_of_starting_frames):
        self.img = cv2.imread(f"{self.input_dir}/{self.frames[i]}", 1)
    
        self.first_frame1 = self.img[:,:,0] 
        self.first_frame2 = self.img[:,:,1] 
        self.first_frame3 = self.img[:,:,2] 

        for h in range(self.height):
            for w in range(self.width):
                self.prev_bg_model1[h][w] = self.prev_bg_model1[h][w] + self.gaussian(self.first_frame1[h][w], b=self.Bd)
                self.prev_bg_model2[h][w] = self.prev_bg_model2[h][w] + self.gaussian(self.first_frame2[h][w], b=self.Bd)
                self.prev_bg_model3[h][w] = self.prev_bg_model3[h][w] + self.gaussian(self.first_frame3[h][w], b=self.Bd)


    self.prev_bg_model1 = self.prev_bg_model1 / num_of_starting_frames
    self.prev_bg_model2 = self.prev_bg_model2 / num_of_starting_frames
    self.prev_bg_model3 = self.prev_bg_model3 / num_of_starting_frames

    self.frames = self.frames


  def process_video(self):
    for I in range(0, len(self.frames)):
      self.tic = time.time()
      self.cnt = self.cnt + 1
      self.gt = 300 * 2 / (1 + exp(-1*(self.cnt - 100)/20))
      self.next_bg_model1 = np.zeros((self.height, self.width, self.Nd))
      self.next_bg_model2 = np.zeros((self.height, self.width, self.Nd))
      self.next_bg_model3 = np.zeros((self.height, self.width, self.Nd))

      self.next_img = cv2.imread(self.input_dir + "/" + self.frames[I], 1)
      self.next_frame1 = self.next_img[:,:,0] 
      self.next_frame2 = self.next_img[:,:,1] 
      self.next_frame3 = self.next_img[:,:,2] 

      self.output_frame = np.zeros((self.height, self.width)) 

      self.process_frame(I)
      if not os.path.isdir(f"output"):
        os.mkdir(f"output")
      
      # self.clean_image()
      cv2.imwrite(f"output/" + self.frames[I], self.output_frame)

      self.prev_bg_model1 = self.next_bg_model1
      self.prev_bg_model2 = self.next_bg_model2
      self.prev_bg_model3 = self.next_bg_model3
      self.toc = time.time()
      print(f"Frame {I+1} done in {int(self.toc-self.tic)} seconds")


  def process_frame(self, I):
      self.threads = []
    #   for h in range(self.height):
    #     self.solve_one_row(h)
      for w in range(self.width):
        self.solve_one_row(w)
      for thread in self.threads:
        thread.join()


  def doit(self):
    self.init_bg_model()
    self.process_video()


  def solve_one_row(self, w):
      for h in range(self.height):
        dist1 = np.min(np.where(self.prev_bg_model1[h][w] > (1/self.Nd), abs(self.x_axis-self.next_frame1[h][w]), 10000))
        dist2 = np.min(np.where(self.prev_bg_model2[h][w] > (1/self.Nd), abs(self.x_axis-self.next_frame2[h][w]), 10000))
        dist3 = np.min(np.where(self.prev_bg_model3[h][w] > (1/self.Nd), abs(self.x_axis-self.next_frame3[h][w]), 10000))


        l = (dist1 / (1 +self.grad1[h][w])) + (dist2 / (1 +self.grad2[h][w])) + (dist3 / (1 +self.grad3[h][w]))
        r = 3 * self.Bd
        if l > r:
          self.grad1[h][w] = (self.gt - 1) * self.grad1[h][w] / self.gt + 0.1 * dist1 / self.gt
          self.grad2[h][w] = (self.gt - 1) * self.grad2[h][w] / self.gt + 0.1 * dist2 / self.gt
          self.grad3[h][w] = (self.gt - 1) * self.grad3[h][w] / self.gt + 0.1 * dist3 / self.gt
          self.output_frame[h][w] = 255
        else:
          self.grad1[h][w] = (self.gt - 1) * self.grad1[h][w] / self.gt + dist1 / self.gt 
          self.grad2[h][w] = (self.gt - 1) * self.grad2[h][w] / self.gt + dist2 / self.gt 
          self.grad3[h][w] = (self.gt - 1) * self.grad3[h][w] / self.gt + dist3 / self.gt 


        self.next_bg_model1[h][w] = self.gaussian(self.next_frame1[h][w], b=self.Bd) / self.gt + 0.95 * self.prev_bg_model1[h][w]
        self.next_bg_model1[h][w] = self.next_bg_model1[h][w] / np.sum(self.next_bg_model1[h][w])

        self.next_bg_model2[h][w] = self.gaussian(self.next_frame2[h][w], b=self.Bd) / self.gt + 0.95 * self.prev_bg_model2[h][w]
        self.next_bg_model2[h][w] = self.next_bg_model2[h][w] / np.sum(self.next_bg_model2[h][w])

        self.next_bg_model3[h][w] = self.gaussian(self.next_frame3[h][w], b=self.Bd) / self.gt + 0.95 * self.prev_bg_model3[h][w]
        self.next_bg_model3[h][w] = self.next_bg_model3[h][w] / np.sum(self.next_bg_model3[h][w])
  

  def clean_image(self):

    mask = np.where(self.output_frame ==0 , 0, 1)

    ii = np.zeros((self.height, self.width)) 

    # base cases 
    ii[0][0] =mask[0][0]
    for w in range(1, self.width):
        ii[0][w] = mask[0][w] + ii[0][w-1]
    for h in range(1, self.height):
        ii[h][0] = mask[h][0] + ii[h-1][0]

    for h in range(1, self.height):
        for w in range(1, self.width):
            ii[h][w] = ii[h-1][w] + ii[h][w-1] - ii[h-1][w-1] + mask[h][w] 

    # for h in range(0, self.height, self.big_patch):
    #     for w in range(0, self.width, self.big_patch):
    #         x1 = h 
    #         y1 = w 
    #         x2 = h+self.big_patch-1
    #         y2 = w+self.big_patch-1
    #         if x2>=self.height: x2=self.height-1
    #         if y2>=self.width: y2=self.width-1

    #         abcd = ii[x2][y2]
    #         a = ii[x1-1][y1-1]
    #         ab = ii[x1-1][y2]
    #         ac = ii[x2][y1-1]
            
    #         if x1-1<0 and y1-1<0:
    #             a = 0
    #             ab = 0
    #             ac = 0
    #         if x1-1<0:
    #             a = 0
    #             ab = 0
    #         if y1-1<0:
    #             a = 0
    #             ac = 0

    #         diff = abcd - ab - ac + a

    #         if diff>=self.big_patch*self.big_patch * 80 / 100:
    #             self.output_frame[x1:x2+1, y1:y2+1] = 255



    for h in range(0, self.height, self.small_patch):
        for w in range(0, self.width, self.small_patch):
            x1 = h 
            y1 = w 
            x2 = h+self.small_patch-1
            y2 = w+self.small_patch-1
            if x2>=self.height: x2=self.height-1
            if y2>=self.width: y2=self.width-1

            abcd = ii[x2][y2]
            a = ii[x1-1][y1-1]
            ab = ii[x1-1][y2]
            ac = ii[x2][y1-1]
            
            if x1-1<0 and y1-1<0:
                a = 0
                ab = 0
                ac = 0
            if x1-1<0:
                a = 0
                ab = 0
            if y1-1<0:
                a = 0
                ac = 0

            diff = abcd - ab - ac + a

            if diff>=self.small_patch*self.small_patch * 3 / 4:
                self.output_frame[x1:x2+1, y1:y2+1] = 255
            elif diff<=self.small_patch*self.small_patch * 1 / 4:
                self.output_frame[x1:x2+1, y1:y2+1] = 0
                

def start(dir_name):
  BGSubtraction(dir_name, 50, 3)

if __name__=='__main__':
  
  # for i in range(1,6):
  #   threadObj = threading.Thread(target=start, args=['input'+str(i)])
  #   threadObj.start() 
  start('input')

import numpy as np
import cv2, os

class Cleaner():
    def __init__(self, input_dir, small_patch, big_patch):
        self.small_patch = small_patch
        self.big_patch = big_patch
        self.input_dir = input_dir
        self.frames = os.listdir(input_dir)
        self.height, self.width = cv2.imread(os.path.join(self.input_dir, self.frames[0]))[:,:,0].shape


        if not os.path.isdir(os.path.join('cleaned_output')):
            os.mkdir('cleaned_output')
        for index, frame in enumerate(self.frames):
            self.output_frame = cv2.imread(os.path.join(self.input_dir, frame))[:,:,0]
            self.clean_image()         

            cv2.imwrite(os.path.join('cleaned_output', frame), self.output_frame)

    def clean_image(self):

        mask = np.where(self.output_frame==0 , 0, 1)

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

        for h in range(0, self.height, self.big_patch):
            for w in range(0, self.width, self.big_patch):
                x1 = h 
                y1 = w 
                x2 = h+self.big_patch-1
                y2 = w+self.big_patch-1
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

                if diff>=self.big_patch*self.big_patch * 95 / 100:
                    self.output_frame[x1:x2+1, y1:y2+1] = 255



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
    


if __name__=='__main__':
    output_dir = 'output'

    if not os.path.isdir('output'):
        print('There must be an directory named "output" where the output of background subtraction is stored. But such directory NOT FOUND')
    obj = Cleaner('output', 3, 17) 

    


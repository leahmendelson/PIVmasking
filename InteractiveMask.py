import numpy as np
import os
from sys import exit
import glob
import cv2
from matplotlib import pyplot as plt
from os.path import basename
from math import pow
import warnings
import time

#warnings.simplefilter(action = "ignore", category = FutureWarning)

#override matplotlib keyboard save
plt.rcParams['keymap.save']=''

#VARIABLES TO CONFIGURE
start_mask = 0# first image to segment (zero-indexed)
last_mask = 50 # last image that needs masking (zero-indexed)
override = 0 # whether to override old mask images

# set up paths
# datapath: input directory of images
# outpath: output directory of images
datapath = r"SampleImages/FishBody"
outpath = r"SampleImages/FishBodyMasked"

mask = None # pre-initialize mask

def rmvParticles(img,sig,offset,hsize):
    #image preprocessing function to suppress bright particles
    iblur = cv2.GaussianBlur(img,(hsize,hsize),sig)
    iblur = iblur + offset

    pixmsk = np.where((img > iblur),1,0).astype('uint8')
    idiff = img-iblur
    ipart = idiff*pixmsk
    idim = img-ipart

    iblur = cv2.GaussianBlur(idim,(hsize,hsize),sig)
    iblur = iblur + offset
    pixmsk = np.where((img > iblur),1,0).astype('uint8')
    idiff = img-iblur
    ipart2 = idiff*pixmsk
    isub = img - ipart2

    return isub

def getRect(all_imgs,num_imgs):
    # defines region of interest for mask using a time-average of all images 
    # to show where body is present at any times
    print('in getRect')
    for k in range (0,num_imgs):
        itemp = cv2.imread(all_imgs[k],0)
        
        if k == 0:
            isum = itemp.astype('float64')
        else:
            isum = isum + itemp.astype('float64')
    iavg = isum/num_imgs

    plt.imshow(iavg)
    plt.title('Click top left and bottom right corners')
    global clickcount
    clickcount = 0

    def corclick(event):   
        print ("in corclick")              
        x = int(event.xdata)
        y = int(event.ydata)
        global clickcount
        global xL, yL, xR, yR, rect
        clickcount = clickcount + 1
        if clickcount == 1:
            print ("Top left selected")
            xL = x
            yL = y
            if yL < 10:
                yL = 0
                
        if clickcount == 2:
            print ("Bottom right selected")
            xR = x
            yR = y
            global rect
            rect = tuple([xL,yL,xR-xL,yR-yL])
            print(rect)
            plt.close(event.canvas.figure)
            
    plt.show()
    cidr = plt.gcf().canvas.mpl_connect('button_press_event', corclick)  
     
    while clickcount < 2:
        plt.pause(1)
        
    return rect
    
 
if __name__ == "__main__":    

    # find files to use
    out_pyr = outpath
    factor = 2 #use pyrDown to improve smoothness & speed processing
    # Scales image size down by factor
                
    # Get all images in each camera folder. This is not natural order sorting.
    all_imgs = sorted(glob.glob(datapath+'/*.tif'))
    use_imgs = all_imgs[start_mask:last_mask]
    num_imgs = len(use_imgs)

    # Check to make sure images were found
    if num_imgs == 0:
        print('No images found. Exiting. Check the path and image ranges.')
        exit()

    savepath = outpath
    if not os.path.exists(savepath):
        os.makedirs(savepath)             

    for j in range(start_mask,last_mask):
        filename = basename(all_imgs[j])
        savefish = savepath+'/'+filename

        if not os.path.isfile(savefish):
            fish_img = cv2.imread(all_imgs[j],0)
            fish_img = cv2.pyrDown(fish_img)

            print("Removing Particles")        
            fish_img = rmvParticles(fish_img,1.5,0,7)
                #fish_show = np.copy(fish_img)
                #cv2.rectangle(fish_show,(rect[0],rect[1]),(rect[0]+rect[2],rect[1]+rect[3]),200,5)
                #plt.imshow(fish_show)
                #plt.show()

            # contrast enhancement once particles are removed
            # can be replaced with other preprocessing options
            clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(16,16))
            fish_img = clahe.apply(fish_img)

            fish_img = cv2.cvtColor(fish_img, cv2.COLOR_GRAY2RGB)

            if j == 0 or mask == None or fgdModel[0,0]==0:
                print("Initializing Mask")
                rectflag = 1
                rect = getRect(use_imgs,num_imgs)
                print('out of getrect')
                print (rect)
                rect=tuple(int(np.around(x/factor)) for x in rect)
                    
                mask = np.zeros(fish_img.shape[:2],np.uint8)
                bgdModel = np.zeros((1,65),np.float64)
                fgdModel = np.zeros((1,65),np.float64)
            else:
                print("Starting with Previous")
                probfish = np.where(mask==0)
                probmask = np.where(mask==1)
                mask[probfish] = 2
                mask[probmask] = 3
                mrect = np.zeros(mask.shape[:2],np.uint8)
                mrect[int(rect[1]):int(rect[1]+rect[3]),int(rect[0]):int(rect[0]+rect[2])]=mask[int(rect[1]):int(rect[1]+rect[3]),int(rect[0]):int(rect[0]+rect[2])]
                mask = mrect
                    
                rectflag = 0
                
            clickflag = 0
            xm = []
            ym = []
            xf = []
            yf = []

            if rectflag:
                print("Running graph cut from RECT")
                cv2.grabCut(fish_img,mask,rect,bgdModel,fgdModel,3,cv2.GC_INIT_WITH_RECT)
            else:
                print("Running graph cut from MASK")                   
                cv2.grabCut(fish_img,mask,rect,bgdModel,fgdModel,3,cv2.GC_INIT_WITH_MASK)

            isfish = np.where((mask==2)|(mask==0),0,1).astype('uint8')
            isbgd = np.where((mask==1)|(mask==3),0,1).astype('uint8')

            img_cut = fish_img*isbgd[:,:,np.newaxis]

            isfish = 255*isfish

            fig = plt.figure()
            axL=plt.subplot(2,2,1)
            axL.imshow(cv2.bitwise_and(fish_img,fish_img,mask = isfish))
            plt.title('Click overmasked regions then hit [D] to update, [C] to clear, or [S] to save')
            axR=plt.subplot(2,2,2)
            plt.title('Click undermasked regions then hit [D] to update, [c] to clear, or [S] to save')
            axR.imshow(fish_img)
            axBL = plt.subplot(2,2,3)
            mskview = axBL.imshow(mask)
            cbar = fig.colorbar(mskview,ticks = [0,1,2,3])
            cbar.set_ticklabels(['Not Fish','Fish','Probs Not Fish','Probs Fish'])
            mng = plt.get_current_fig_manager()
            mng.full_screen_toggle()

            def save_figure(event):
                if event.key == 's':
                    plt.close(event.canvas.figure)
                    print("Rescaling fish")
                    global isfish
                    isfish = cv2.resize(isfish,None,fx=factor, fy=factor, interpolation = cv2.INTER_CUBIC)
                    print(savefish)
                    cv2.imwrite(savefish,isfish)

            def done(event):
                if event.key == 'd':

                    global xm
                    global ym
                    global xf
                    global yf
                    print('Updating mask')
                    mask[ym,xm]=0
                    mask[yf,xf]=1
                    axBL.imshow(mask)
                    fig.canvas.draw()

                    print('Revising graph cut')
                    cv2.grabCut(fish_img,mask,rect,bgdModel,fgdModel,3,cv2.GC_INIT_WITH_MASK)
                    global isfish
                    isfish = np.where((mask==2)|(mask==0),0,1).astype('uint8')
                    isbgd = np.where((mask==1)|(mask==3),0,1).astype('uint8')

                    img_cut = fish_img*isbgd[:,:,np.newaxis]

                    isfish = 255*isfish
                    axL.imshow(cv2.bitwise_and(fish_img,fish_img,mask = isfish))
                    #plt.title('Click overmasked regions or hit [S] to save')
                    #plt.title('Click undermasked regions or hit [S] to save')
                    axR.imshow(fish_img)
                    axBL.imshow(mask)
                    print('Redrawing mask')
                    fig.canvas.draw()

            def clear(event):
                if event.key == 'c':
                    global xm
                    global ym
                    global xf
                    global yf
                    global mask
                    print('Resetting mask')

                    mask = np.zeros(fish_img.shape[:2],np.uint8)
                    bgdModel = np.zeros((1,65),np.float64)
                    fgdModel = np.zeros((1,65),np.float64)

                    print('Restarting graph cut')
                    cv2.grabCut(fish_img,mask,rect,bgdModel,fgdModel,3,cv2.GC_INIT_WITH_RECT)
                    global isfish
                    isfish = np.where((mask==2)|(mask==0),0,1).astype('uint8')
                    isbgd = np.where((mask==1)|(mask==3),0,1).astype('uint8')

                    clickflag = 0
                    xm = []
                    ym = []
                    xf = []
                    yf = []

                    img_cut = fish_img*isbgd[:,:,np.newaxis]

                    isfish = 255*isfish
                    axL.imshow(cv2.bitwise_and(fish_img,fish_img,mask = isfish))
                    #plt.title('Click overmasked regions or hit [S] to save')
                    #plt.title('Click undermasked regions or hit [S] to save')
                    axR.imshow(fish_img)
                    axBL.imshow(mask)
                    print('Redrawing mask')
                    fig.canvas.draw()


            def onclick(event):
                global clickflag
                clickflag = 1
                x = int(event.xdata)
                y = int(event.ydata)
                if event.inaxes == axL:
                    ax = 0
                    xm.append(x)
                    ym.append(y)
                elif event.inaxes == axR:
                    ax = 1
                    xf.append(x)
                    yf.append(y)

            def drag(event):
                global clickflag
                if clickflag == 1:
                    if event.xdata is not None:
                        x = int(event.xdata)
                        y = int(event.ydata)
                        if event.inaxes == axL:
                            ax = 0
                            xm.append(x)
                            ym.append(y)
                        elif event.inaxes == axR:
                            ax = 1
                            xf.append(x)
                            yf.append(y)
                    
            def release(event):
                global clickflag
                clickflag = 0

            cid = plt.gcf().canvas.mpl_connect('key_press_event', save_figure)
            cid2 = plt.gcf().canvas.mpl_connect('button_press_event', onclick)
            cid3 = plt.gcf().canvas.mpl_connect('key_press_event', done)
            cid4 = plt.gcf().canvas.mpl_connect('button_release_event', release)
            cid5 = plt.gcf().canvas.mpl_connect('motion_notify_event',drag)
            cid6 = plt.gcf().canvas.mpl_connect('key_press_event', clear)

            plt.show()
        
            




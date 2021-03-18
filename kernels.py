# lista de alguns kernels
import numpy as np
import cv2
from Op import OpImage

smallBlur = np.ones((7, 7), dtype="float") * (1.0 / (7 * 7))
largeBlur = np.ones((21, 21), dtype="float") * (1.0 / (21 * 21))

sharpen = np.array((
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0]), dtype="int")

laplacian = np.array((
    [0, 1, 0],
    [1, -4, 1],
    [0, 1, 0]), dtype="int")

sobelX = np.array((
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]), dtype="int")

sobelY = np.array((
    [1, 2, 1],
    [0, 0, 0],
    [-1, -2, -1]), dtype="int")

prewittX = np.array((
    [1, 0, -1],
    [1, 0, -1],
    [1, 0, -1]), dtype="int")

prewittY = np.array((
    [1, 1, 1],
    [0, 0, 0],
    [-1, -1, -1]), dtype="int")


edge1 = np.array((
    [1,0,-1],
    [0, 0, 0],
    [-1,0,1]), dtype="int")

edge2 = np.array((
    [0,-1,0],
    [-1, 4, -1],
    [0, -1, 0]), dtype="int")

edge3 = np.array((
    [-1, -1, -1],
    [-1, 8, -1],
    [-1, -1, -1]), dtype="int")

boxblur = np.array((
    [1,1,1],
    [1,1,1],
    [1,1,1]), dtype="int") * (1/9)

gaussblur = np.array((
    [1,2,1],
    [2,4,2],
    [1,2,1]), dtype="int") * (1/16)

unsharpmasking = np.array((
    [1,4,6,4,1],
    [4,16,24,16,4],
    [6,24,-476,24,6],
    [4,16,24,16,4],
    [1,4,6,4,1]), dtype="int") * (-1/256)

kernels_list = [
	#("smallBlur", smallBlur),
    #("largeBlur", largeBlur),
    #("sharpen", sharpen),
    #("laplacian", laplacian),
    ("sobelX", sobelX),
    ("sobelY", sobelY),
    #("edge1",edge1),
    #("edge2",edge2),
    #("edge3",edge3),
    #("boxBlur",boxblur),
    #("gaussBlur",gaussblur),
    #("unsharpMasking",unsharpmasking)
]


def runAllKernels(image):
    for (kernelName, kernel) in kernels_list:
        print("[INFO] applying {} kernel".format(kernelName))
        convoleOutput = OpImage.convolve(image, kernel)
        opencvOutput = cv2.filter2D(image, -1, kernel)
        # show the output images
        cv2.imshow("{} - convole".format(kernelName), convoleOutput)
        cv2.imshow("{} - opencv".format(kernelName), opencvOutput)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def runAllKernelsSimultaneously(image):
    for (kernelName, kernel) in kernels_list:
        print("[INFO] applying {} kernel".format(kernelName))
        convoleOutput = OpImage.convolve(image, kernel)
        #opencvOutput = cv2.filter2D(image, -1, kernel)
        # show the output images
        cv2.imshow("{} - convole".format(kernelName), convoleOutput)
        #cv2.imshow("{} - opencv".format(kernelName), opencvOutput)
        #cv2.destroyAllWindows()
    cv2.waitKey(0)
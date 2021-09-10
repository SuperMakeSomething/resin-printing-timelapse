"""
Resin 3D Printing Timelapse Code - Super Make Something Episode 23
by: Alex - Super Make Something
date: August 29th, 2021
license: Creative Commons - Attribution - Non-Commercial.  More information available at: http://creativecommons.org/licenses/by-nc/3.0/
"""

import cv2
import os
import numpy as np
import subprocess


def ResizeWithAspectRatio(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation=inter)


if __name__ == '__main__':
    # Define program options
    fromCenter = False  # Drag ROI bounding box from top left to bottom right
    imDir = 'Frames/'  # Directory of images
    targetDir = 'Output/'
    if not os.path.exists(imDir + targetDir):  # Create directory for target images if it doesn't exist
        os.makedirs(imDir + targetDir)
    visualizeDir = 'Visualization/'
    if not os.path.exists(imDir + visualizeDir):  # Create directory for visualization images if it doesn't exist
        os.makedirs(imDir + visualizeDir)
    tempDir = 'Temp/'
    if not os.path.exists(imDir + tempDir):  # Create directory for visualization images if it doesn't exist
        os.makedirs(imDir + tempDir)

    #  filenames = os.listdir(imDir)  # Get the contents of imDir
    #  filenames = filter(os.path.isfile, os.listdir(imDir))  # Only keep filenames
    filenames = [f for f in os.listdir(imDir) if os.path.isfile(os.path.join(imDir, f))]  # Get names of files in imDir
    markerLoc = np.empty([len(filenames), 2])  # (x,y) location of markers in images
    imageNames = np.empty(len(filenames), dtype=object)   # Filename of corresponding image

    # Read images
    im = cv2.imread(imDir + filenames[0])  # Read first image

    # Select ROI
    imResize = ResizeWithAspectRatio(im, width=1280)  # Resize image
    roi = cv2.selectROI(imResize, fromCenter)  # Select ROI
    cv2.destroyAllWindows()  # Close window
    imCrop = imResize[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])]  # Crop image to keep only ROI
    # cv2.imshow("Image", imCrop)  # Display ROI crop as sanity check
    # cv2.waitKey(0)  # Wait for keyboard input to continue

    # Find fiducial marker location in all images
    for i in range(len(filenames)):
        print('Processing Image '+ str(i+1) + ' of '+str(len(filenames)))
        im = cv2.imread(imDir + filenames[i])  # Read image
        imResize = ResizeWithAspectRatio(im, width=1280)  # Resize image
        imCrop = imResize # imResize[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])]  # Crop image to keep only ROI
        imGray = cv2.cvtColor(imCrop, cv2.COLOR_BGR2GRAY)  # Convert ROI to grayscale
        # cv2.imshow("Image", imGray)
        # cv2.waitKey(0)
        circles = cv2.HoughCircles(imGray, cv2.HOUGH_GRADIENT, dp=1, minDist=100, param1=20, param2=50, minRadius=10, maxRadius=30)  # Find the circles
        if circles is not None:
            # convert the (x, y) coordinates and radius of the circles to integers
            circles = np.round(circles[0, :]).astype("int")
            # loop over the (x, y) coordinates and radius of the circles
            for (x, y, r) in circles:
                cv2.imwrite(imDir + 'NoCircle/' + filenames[i], imCrop)
                # draw the circle in the output image, then draw a rectangle
                # corresponding to the center of the circle
                cv2.circle(imCrop, (x, y), r, (0, 255, 0), 4)
                cv2.rectangle(imCrop, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
                markerLoc[i, :] = np.array([x, y])  # Save marker (x,y) location
                imageNames[i] = filenames[i]  # Save corresponding image name
                cv2.imwrite(imDir + tempDir + filenames[i], imCrop)
            # show the output image
            # cv2.imshow("Image", imCrop)
            # cv2.waitKey(0)

    np.savez('data', markerLoc=markerLoc, imageNames=imageNames)

    data = np.load('data.npz', allow_pickle=True)
    markerLoc = data.f.markerLoc
    imageNames = data.f.imageNames

    idxSorted = np.argsort(-markerLoc[:, 1], axis=0)  # Sort by y locations of identified markers
    markerLocSorted = markerLoc[idxSorted, :]
    filenamesSorted = imageNames[idxSorted]  # Sort filenames based on y locations of identified markers

    # Remove identifications outside of certain pixel location range y-range
    deleteIdx = np.argwhere(markerLocSorted[:, 1] > 500)
    markerLocSorted = np.delete(markerLocSorted, deleteIdx, axis=0)
    filenamesSorted = np.delete(filenamesSorted, deleteIdx)

    deleteIdx = np.argwhere(markerLocSorted[:, 1] < 1)
    markerLocSorted = np.delete(markerLocSorted, deleteIdx, axis=0)
    filenamesSorted = np.delete(filenamesSorted, deleteIdx)

    deleteIdx = np.argwhere(markerLocSorted[:, 1] > 500)
    markerLocSorted = np.delete(markerLocSorted, deleteIdx, axis=0)
    filenamesSorted = np.delete(filenamesSorted, deleteIdx)

    deleteIdx = np.argwhere(markerLocSorted[:, 1] < 1)
    markerLocSorted = np.delete(markerLocSorted, deleteIdx, axis=0)
    filenamesSorted = np.delete(filenamesSorted, deleteIdx)

    # markerLocSorted = markerLocSorted[markerLocSorted[:,0] !=np.array(None)]
    markerLocSorted = markerLocSorted[~np.isnan(markerLocSorted[:, 0])]
    filenamesSorted = filenamesSorted[filenamesSorted != np.array(None)]  # Remove failed identifications (None) from array

    minYVal = np.amin(markerLocSorted[:, 1])
    maxYVal = np.amax(markerLocSorted[:, 1])

    # filenamesKeep = np.empty(len(filenamesSorted), dtype=object)

    filenamesSorted = filenamesSorted.tolist()
    i = 0
    filenamesKeep = filenamesKeep = []
    markerLocSortedYKeep = []
    for counter in range(int(minYVal), int(maxYVal), 1):

        keepIdx = np.argwhere(markerLocSorted[:, 1] == counter)
        if keepIdx.size != 0:
            filenamesKeep.append(filenamesSorted[int(keepIdx[0])])
            markerLocSortedYKeep.append(markerLocSorted[int(keepIdx[0]), 1])

    # Identify frames for RIFE interpolation
    interpThresh = 2
    differences = np.diff(markerLocSortedYKeep)
    filenamesInterp = [('temp1', 'temp2', 0)]  # Declare list with temporary element
    for i in range(len(differences)):
        if differences[i] > interpThresh:
            filenamesInterp.append((filenamesKeep[i], filenamesKeep[i + 1], differences[i]))

    filenamesInterp.pop(0)  # Remove temporary element

    # filenamesKeep = filenamesKeep.tolist()
    print('Found fiducials in ' + str(np.shape(filenamesSorted)[0]) + ' images.')
    print('Keeping ' + str(np.shape(filenamesKeep)[0]) + ' images.')

    # RIFE interpolate files
    for i in range(len(filenamesInterp)):

        filenameA = 'Frames/' + filenamesInterp[i][0]
        filenameB = 'Frames/' + filenamesInterp[i][1]
        frameDifference = filenamesInterp[i][2]
        multiplier = int(np.sqrt(frameDifference))

        # Call RIFE

        multString = '--exp=' + str(multiplier)
        my_command = ['python', 'inference_img.py', '--img', filenameA, filenameB, multString]

        # my_env = os.environ.copy()
        # my_env["PATH"] = "/usr/sbin:/sbin:" + my_env["PATH"]
        process = subprocess.Popen(my_command, shell=True)
        print("--------------------")
        print("Processing interpolation " + str(i + 1) + " of " + str(len(filenamesInterp)) + ".")
        print("Generating " + str(multiplier ** 2) + " frames between " + filenameA + " and " + filenameB + ".")
        process.wait()
        print("Interpolation " + str(i + 1) + " completed.")

        print("Renaming and moving files.")
        j = 0
        for file in os.listdir("output"):
            if j == 0:
                os.remove(
                    "output/" + file)  # Delete file -- first frame in the output folder is a copy of the start frame
            else:
                newFilename = filenameA[7:-4] + "_" + str(j).zfill(3) + file[-4:]
                os.rename("output/" + file, "Frames/" + newFilename)  # Rename the file

                # Insert the filename into the correct location of "filenamesKeep" list
                startFrameIdx = filenamesKeep.index(filenameA[7:])  # Find index of starting frame
                filenamesKeep.insert(startFrameIdx + j, newFilename)  # Insert new frame into correct list location

            j = j + 1

    # Resave images in sorted order
    for i in range(len(filenamesKeep)):
        print('Exporting Image ' + str(i + 1) + ' of ' + str(len(filenamesKeep)))
        imOutput = cv2.imread(imDir + filenamesKeep[i])  # Read image
        cv2.imwrite(imDir + targetDir + str(i).zfill(5) + '.jpg', imOutput)

        # imVisualize = cv2.imread(imDir + tempDir + filenamesKeep[i])
        # cv2.imwrite(imDir + visualizeDir + str(i).zfill(5) + '.jpg', imVisualize)

    print('Done!')

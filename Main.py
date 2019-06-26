import cv2 as cv
import numpy
from imutils import paths

multRostos = 0
umRosto = 0
nenhumRosto = 0
percorridas = 0

for imagePath in paths.list_images("img"):
    # Read image from your local file system
    original_image = cv.imread(imagePath)

    # Convert color image to grayscale for Viola-Jones
    grayscale_image = cv.cvtColor(original_image, cv.COLOR_BGR2GRAY)

    # Load the classifier and create a cascade object for face detection
    face_cascade = cv.CascadeClassifier('haarcascade_frontalface_alt.xml')

    detected_faces = face_cascade.detectMultiScale(grayscale_image)

    for (column, row, width, height) in detected_faces:
            #cv.rectangle(
            #    original_image,
            #    (column, row),
            #    (column + width, row + height),
            #    (0, 255, 0),
            #    2
            #)
        #cv.imshow('Image', original_image)
        #cv.waitKey(0)
        #cv.destroyAllWindows()

        crop_img = original_image[column:column+height, row:row+width]
        cv.imshow("cropped", crop_img)
        cv.waitKey(0)
        cv.destroyAllWindows()

    if len(detected_faces) == 0:
        nenhumRosto += 1
    
    if len(detected_faces) == 1:
        umRosto += 1        
    
    if len(detected_faces) > 1:
        multRostos += 1

    percorridas += 1
    print('imagens percorridas', str(percorridas))

print('Imagens com um rosto encontrado: ', str(umRosto))
print('Imagens com mais de um rosto encontrado: ', str(multRostos))
print('Imagens sem nenhum rosto encontrado: ', str(nenhumRosto))
        
        #print('encontrou')
        #print(detected_faces)

        #for (column, row, width, height) in detected_faces:
        #    cv.rectangle(
        #        original_image,
        #        (column, row),
        #        (column + width, row + height),
        #        (0, 255, 0),
        #        2
        #    )

        #cv.imshow('Image', original_image)
        #cv.waitKey(0)
        #cv.destroyAllWindows()


        #hog = cv.HOGDescriptor()
        #h = hog.compute(original_image)

        #print(h)

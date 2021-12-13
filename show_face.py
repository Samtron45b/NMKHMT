import numpy as np
import cv2
import pickle
import tensorflow as tf
from keras.preprocessing.image import img_to_array



SVMModel, SNNModel, LeNetModel = None, None, None



def SVM_smile(npimage: np.ndarray):
    return SVMModel.predict(npimage)



def SqNN_smile(npimage: np.ndarray):
    npimage = npimage.reshape(1, -1)
    result = SNNModel.predict(npimage)
    return np.argmax(result, axis=1)



def LeNet_smile(npimage: np.ndarray):
    roi = cv2.resize(npimage, (28, 28))
    roi = roi.astype("float") / 255.0
    roi = img_to_array(roi)
    roi = np.expand_dims(roi, axis=0)
    result = LeNetModel.predict(roi)
    return np.argmax(result, axis=1)



def preprocessing_image(image):
    IMG_SIZE = 50
    test_face_gray = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    test_face_gray = np.array(test_face_gray)
    test_face_gray = test_face_gray/255.0
    test_face_gray = test_face_gray.flatten().reshape(-1, IMG_SIZE*IMG_SIZE)
    return test_face_gray



def capturing_from_webcam(algorithm):
    #the video capturer:
    vidcap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    
    #face detector:
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    

    while True:
        #get image from the captured video:
        ret, frame = vidcap.read()


        #gray scale image to make things eassier:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        

        #detect location of the face:
        faces = face_cascade.detectMultiScale(gray, 1.05, 5)


        for (x, y, w, h) in faces:
            #draw the rectangle(s) surround(s) each detected human face:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 5)

            if (algorithm != "None"):
                top, left, bottom, right = y, x, y + h, x + w
                #get face only:
                face_gray = gray[top:bottom, left:right]

                #preprocessing image:
                test_face_gray = preprocessing_image(face_gray)

                #some needed variables to check the state of the detected face:
                isSmiling, state = 0, "Not smiling"

                #choose the algorithm to make a smile detection:
                if (algorithm == "SVM"): isSmiling = SVM_smile(test_face_gray)
                elif (algorithm == "SqNN"): isSmiling = SqNN_smile(test_face_gray)
                elif (algorithm == "LeNet"): isSmiling = LeNet_smile(face_gray)
                if (isSmiling == 1): state = "Smiling"

                #draw the text "Smiling"/"Not smiling" with box for each detected human face:
                #box:
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (225, 0, 0), cv2.FILLED)
                #text:
                cv2.putText(frame, state, (left + 5, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 1.0, 
                            (225, 225, 225), 2)


        #create window to show captured image on the screen screen
        cv2.imshow('Captured face', frame)


        #make program wait for 1 milisecond in each loop
        #and it will stop the process immediately after user press 'q' on keyboard
        if (cv2.waitKey(1) & 0xFF == ord('q')):
            break


    #clearing everything before end
    vidcap.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    last_choice = ". None (Only show the face video with face detection"
    choice_menu = "0. Exit\n1. SVM\n2. Sequential Neural Network\n3. LeNet Convolutional Network"
    num_choices = len(choice_menu.split("\n"))
    choice_menu = str(choice_menu + "\n" + str(num_choices) + last_choice)
    choice, algorithm = -1, ""
    while (True):
        num_choices = len(choice_menu.split("\n"))
        print("Choice menu:")
        print(choice_menu)
        choice = input("Choose the algorithm you want to use: ")
        if (choice < "0" or choice > str(num_choices)):
            print("Invalid input\n\n")
        else:
            if (choice == "0"): break
            elif (choice == "1"):
                if (SVMModel == None):
                    with open("./models/SVM/svm_model.sav", "rb") as content:
                        SVMModel = pickle.load(content)
                algorithm = "SVM"
            elif (choice == "2"):
                if (SNNModel == None):
                    SNNModel = tf.keras.models.load_model("./models/SNN")
                algorithm = "SqNN"
            elif (choice == "3"): 
                if (LeNetModel == None):
                    LeNetModel = tf.keras.models.load_model("./models/LeNet")
                algorithm = "LeNet"
            elif (choice == "4"): algorithm = "None"
            capturing_from_webcam(algorithm)
        print()
    print("Nothing wrong happenned. Good bye!\n")

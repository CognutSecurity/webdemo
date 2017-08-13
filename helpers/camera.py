import os 
import time
import threading
class Camera(object):
    def __init__(self, index, width, height):
        # noinspection PyArgumentList
        self.camera = cv2.VideoCapture(index)

        self.camera.set(3, width)
        self.camera.set(4, height)

    def getBase64Image(self):
        _, frame = self.camera.read()

        _, image = cv2.imencode(".jpg", frame)

        return base64.b64encode(image.tostring())

    def saveImageToFile(self, name, folder="."):
        _, frame = self.camera.read()

        cv2.imwrite("%s/%s.jpg" % (folder, name), frame)


def removeOldImages(interval, folder, old):
    for filename in os.listdir(folder):
        image = os.path.join(folder, filename)

        if os.path.isfile(image) and os.stat(image).st_mtime < time.time() - old:
            os.unlink(image)

    threading.Timer(interval, removeOldImages, [interval, folder, old]).start()

def storeImageByTimer(webCamera, interval, folder):
    webCamera.saveImageToFile(time.strftime("%d-%m-%Y_%H-%M-%S"), folder)

    threading.Timer(interval, storeImageByTimer, [webCamera, interval, folder]).start()


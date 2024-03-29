import time
import threading

class SetInterval:
    def __init__(self,interval, action):
        self.interval = interval
        self.action = action
        self.stopEvent = threading.Event()
        self.thread = threading.Thread(target=self.__setInterval)

    def get_thread(self):
        return self.thread
    
    def start(self):
        self.thread.start()

    def __setInterval(self):
        nextTime = time.time()+self.interval
        while not self.stopEvent.wait(nextTime-time.time()):
            nextTime += self.interval
            self.action()

    def stop(self):
        self.stopEvent.set()

import threading, queue
import time
import zmq
import logging
import sys
import json


from .basic_thread import BasicThread

class Collector(BasicThread):
    ''' RequestHandler will receive requests from MPQUIC
        Pass on the requests to the CentralTrainer
        Obtain a response (scheduling-related) and send it back
    '''
    def __init__(self, threadID: int, threadName: str, queue: queue.Queue, host:str="localhost", port:str="5556"):
        super().__init__(threadID, threadName, queue)

        # Stream times
        self._all_streams = []

        # ZMQ context
        self.__host = host
        self.__port = port
        
        self.__context = zmq.Context()
        self._subscriber = self.__context.socket(zmq.SUB)
        self._subscriber.connect("tcp://%s:%s" % (self.__host, self.__port))
        self._subscriber.subscribe("")

        self.__poller = zmq.Poller()
        self.__poller.register(self._subscriber, zmq.POLLIN)

    def start(self):
        super().start()

    def run(self):
        self.pinfo("Run Collector Thread")
        while not self._stoprequest.isSet():
            try:
                # Poll for a reply => time is ms
                if (self.__poller.poll(timeout=10)):
                    # Receive request from middleware
                    try:
                        data = self._subscriber.recv_multipart(zmq.NOBLOCK)
                        #data = self._subscriber.recv(zmq.NOBLOCK)
                    except Exception as ex:
                        self.pdebug(ex)
                        continue

                    json_data = json.loads(data[1])
                    #print(json_data)
                    self.pinfo(json_data)
                    keys = ["segment_nr",
                    "bandwidth",
                    "bitrate",
                    "down_shifts",
                    "buffering_ratio",
                    "initial_buffering",
                    "stall_count",
                    "stall_sum",
                    "qoe"]
                    json_data = {k: json_data[k] for k in keys}
                    self._all_streams.append(json_data)
                    
                    # put stream info on the Queue (blocking operation)
                    self.putrequest(json_data)
            except Exception as ex:
                self.pdebug(ex)
        self.close()

    def close(self):
        self._subscriber.close()
        self.__context.term()
        self.pinfo("RequestHandler closing gracefully...")



if __name__ == "__main__":
    tqueue = queue.Queue()

    cthread = Collector(1, 'collector', queue=tqueue)
    cthread.start()
    cthread.join()
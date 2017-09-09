import time


class Timer:

    def __init__(self):
        self.__start_stamp = 0
        self.__last_stamp = 0
        self.__step = 0

    def start(self):
        self.__start_stamp = time.time()
        self.__last_stamp = self.__start_stamp
        self.__step = 0

    def tick(self, show_full_time_spend=False):
        self.__step += 1
        if show_full_time_spend:
            print("Step%s : full_spend %sms" % (str(self.__step), str(
                int((time.time() - self.__start_stamp) * 1000))))
        else:
            print("Step%s : from last step spent %sms" % (
                str(self.__step), str(int((time.time() - self.__last_stamp) * 1000))))
            self.__last_stamp = time.time()

    def flush(self):
        self.__start_stamp = time.time()
        self.__last_stamp = self.__start_stamp
        self.__step = 0

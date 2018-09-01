import os
import time

class Reporter():
    '''
    Each time report is called it counts one more loop. Also, if
    enough time has passed since last report (sleep), reports
    progress unsing total_loops.
    total_loops -- int
    msg         -- str
    sleep       -- float (in seconds)
    '''
    def __init__(self, total_loops, msg = None, sleep = 0.5):
        '''
        total_loops: ing with total number of svgs to be
                     converted
                msg: str, before the % of progress
              sleep: seconds between prints
        '''
        self.total_loops = total_loops
        self.sleep = sleep
        self.last_time = 0
        self.progress = 0
        if msg is None:
            self.msg = '\rProgress: '
        else:
            self.msg = '\r' + msg + ': '
    def report(self, end = False):
        # Call to get a report on progress
        self.progress += 1
        if end:
            print(self.msg + 'Completed')
        elif time.time() - self.last_time > self.sleep:
            x = round(100*self.progress/self.total_loops,2)
            print(self.msg +'{:3.2f} %'.format(x),end='')
            self.last_time = time.time()
            
def get_video_paths(dir_path):
    '''
    Gets paths of all video files in the subdirs of dir 
    '''
    video_paths = []
    for d in os.listdir(dir_path):
        sub_path = os.path.join(dir_path, d)
        if os.path.isdir(sub_path):
            for i in os.listdir(sub_path):
                if i.endswith('.mkv') or i.endswith('.mp4'):
                    video_paths.append(os.path.join(sub_path,i))
    return video_paths
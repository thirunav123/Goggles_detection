import sys,time,os
from matplotlib import pyplot as plt

from PyQt5.uic import loadUi
from PyQt5 import QtWidgets,QtCore
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import QApplication, QGraphicsDropShadowEffect
from PyQt5.QtCore import Qt
APP_FOLDER = 'D:\I4.0\Goggle_detection\File_access_test\Detection_records'

class loadingWindow(QtWidgets.QMainWindow):
    def __init__(self):
        
        super( loadingWindow, self).__init__()
        loadUi('page2.ui', self)
        self.violation_graph.clicked.connect(self.show_graph)
        self.logout.clicked.connect(self.close)
        self.show()
        fig= plt.figure()
        fig.patch.set_facecolor('gray')
        fig.patch.set_alpha(0.5)        
    def show_graph(self):
        totalFiles = 0
        totalDir = 0
        count=0
        asdf=[]
        max_recent_files_read=4
        x=[]
        y=[]
        # self.figure=plt.figure()
        for base, dirs, files in os.walk(APP_FOLDER):
            # print(base)
            # print('Searching in : ',base)
            # paths = sorted(Path(base).iterdir(), key=os.path.getmtime)
            # print(paths)
            for directories in dirs:
                # for 
                asdf.append(os.path.join(base,directories))
                        
                # print(directories)
                totalDir += 1
            # for Files in files:
                # asd=os.path.join(base,Files)
                # print(date.fromtimestamp(os.stat(asd).st_mtime))
                # totalFiles += 1
            # print('Total number of files',totalFiles)
            # print('Total Number of directories',totalDir)
            # print('Total:',(totalDir + totalFiles))
            count=count+1
            # print(asd)
            asdf.sort(key=os.path.getctime,reverse=1)
            for i in range(len(asdf)):
                totalFiles=0
                # print("asd")
                # print(asdf[i])
                for B, D, F in os.walk(asdf[i]):
                    # print(B)
                    # break
                    # print(F)
                    for a in F:
                # asd=os.path.join(base,Files)
                # print(date.fromtimestamp(os.stat(asd).st_mtime))
                        totalFiles += 1
                    break
                x.append(os.path.split(asdf[i])[1])
                y.append(totalFiles)
            print(x)
            print(y)
            plt.clf()
            plt.title("Violation Graph")
            plt.xlabel("Date")
            plt.ylabel("No of frames")
            plt.xlim(len(x)-5-0.5,len(x))
            # for i in range(len(x)):
            #     plt.annotate(i,y[i],y[i],ha="center")
            # self.figure.patch.set_facecolor('blue')
            # self.figure.patch.set_alpha(0.5)
            plots=plt.bar(x,y)
            # plt.
            # plt.figure().na=
            # if count=

            for bar in plots:
                # print(bar.get_height())
                height=bar.get_height()
                plt.annotate('{}'.format(height),
                xy=(bar.get_x()+bar.get_width()/2,height),
                xytext=(0,2),
                textcoords='offset points',ha='center',va='center')

            # if count==1:
            break
        plt.show()
    
    def close(self):
        plt.close()


app = QApplication(sys.argv)
movex_limit=app.desktop().screenGeometry().width()
movey_limit=app.desktop().screenGeometry().height()
# print(QDesktopWidget().availableGeometry().width())
ex = loadingWindow()
# ex.show()
# ex.progress(6)
# time.sleep(5)
app.exec_()
print("completed")

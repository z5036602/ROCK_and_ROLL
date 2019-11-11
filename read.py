import pandas as pd
import os.path

# needs to put this file under sensing folder
class Attr:
   
    def __init__(self):
        self.activity = []
        i = 0
        while (i < 60):
            if (i < 10 and os.path.exists("activity/activity_u0"+str(i)+".csv")):
                self.activity.append(pd.read_csv("activity/activity_u0"+str(i)+".csv"))
            
            if (i >= 10 and os.path.exists("activity/activity_u"+str(i)+".csv")):
                self.activity.append(pd.read_csv("activity/activity_u"+str(i)+".csv"))

            i = i + 1

    def display(self):
        print(self.activity[0])

a = Attr()
a.display()


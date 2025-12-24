# Step 1: Importing required modules
import pandas as pd
import numpy as np
import matplotlib as plt

#Step 2 : feeding Dataset 
df = pd.read_csv('Placement_BeginnerTask01.csv')
#checking if it is reading by print(dataset.head(4))

#Step 3 : Remove not required data
df = df.drop('StudentID',axis=1) #axis=1 is column and 0 is row

#step 4 categorizing data 
df["ExtracurricularActivities"] = df["ExtracurricularActivities"].astype('category') #.astype() converts datatypes (numbers,float,int,text,object,category)
df["ExtracurricularActivities"] = df["ExtracurricularActivities"].cat.codes   #cat.codes converts categories assingning numbers

df["PlacementTraining"]= df["PlacementTraining"].astype('category')
df["PlacementTraining"] = df["PlacementTraining"].cat.codes


df["PlacementStatus"]= df["PlacementStatus"].astype('category')
df["PlacementStatus"] = df["PlacementStatus"].cat.codes

#step 5 seperating data as X(inputs) and Y(output values)
X = df.iloc[:, :-1].values #slicing all rows and all columns except last column(-1)
Y = df.iloc[:, -1].values

#step 6 spliting dataset into training and testing inputs and outputs

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2 )

# remember order is x(train),x(test),y(train),y(test) if used X,Y then test size 0.2 means 20% random data is split for testing and remaining for training.

#step 7 creating a classifier using scikit lib
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(random_state=0, solver='lbfgs', # clf - classifyer
                         max_iter=1000).fit(X_train, 
                                            Y_train)
# step 8 printing the acc
clf.score(X_test, Y_test)

#step 9 get new input from users for predicting based on learning by model 
a = float(input("Enter CGPA: "))
b = int(input("Enter number of internships: "))
c = int(input("Enter the number of projects: "))
d = int(input("Enter number of workshops; "))
e = float(input("Enter aptitude test score (out of 100): "))
f = float(input("Enter soft skill score (out of 5): "))
g = int(input("Participation in Extracuricular activities,number(Yes(1)/No(0)): "))
h = int(input("Completed placement training,number(Yes(1)/No(0)): "))
i = int(input("Enter SSC marks: "))
j = int(input("Enter HSC Marks: "))

# step 10 predicting for new inputs
#order: CGPA,Internships,projects,workshops,apti test, soft skill score, extrac activities, placementtraining, ssc_marks,hsc_marks,placement status

z = clf.predict([[a,b,c,d,e,f,g,h,i,j]])
if z == 1:
    print("May get placed")
else: print("May not get placed")
    
print(clf.predict_proba([[a,b,c,d,e,f,g,h,i,j]]))
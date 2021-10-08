import csv
import numpy as np


class TitanicData():
    def __init__(self,train_data_path,test_data_path,test_label_path):
        self.train_data_path = train_data_path
        self.test_data_path = test_data_path
        self.test_label_path = test_label_path
        
        self.train_data_reader = csv.reader(open(train_data_path,"r"))
        self.test_data_reader = csv.reader(open(test_data_path,"r"))
        self.test_label_reader = csv.reader(open(test_label_path,"r"))
        
        self.train_data_dict = self.Loading_Train_Data()
        self.test_data_dict = self.Loading_Test_Data()

    def Loading_Train_Data(self):
        train_data = {}
        for line in self.train_data_reader:
            if line[0] == "PassengerId" : 
                train_data["Title"] = line
            else:
                PassengerId = int(line[0])
                train_data[PassengerId] = {}
                for index,title in enumerate(train_data["Title"]):
                    if index == 0 : continue
                    elif title == "Survived" : 
                        train_data[PassengerId][title] = int(line[index])
                    elif title == "Pclass" : 
                        train_data[PassengerId][title] = int(line[index])
                    elif title == "Sex":
                        if line[index] == "male":
                            train_data[PassengerId][title] = 1
                        else:
                            train_data[PassengerId][title] = -1
                    elif title == "Age":
                        train_data[PassengerId][title] = float(line[index]) if line[index] != "" else 0.0
                    elif title == "SibSp":
                        train_data[PassengerId][title] = int(line[index])
                    elif title == "Parch":
                        train_data[PassengerId][title] = int(line[index])
                    elif title == "Fare":
                        if line[index] != "":
                            train_data[PassengerId][title] = float(line[index])
                        else:
                            train_data[PassengerId][title] = 0
                    elif title == "Embarked":
                        if line[index] == "S":
                            train_data[PassengerId][title] = 1
                        if line[index] == "C":
                            train_data[PassengerId][title] = 2
                        if line[index] == "Q":
                            train_data[PassengerId][title] = 3
                        else:
                            train_data[PassengerId][title] = 0
                    else:
                        train_data[PassengerId][title] = line[index]
        return(train_data)
    
    def Loading_Test_Data_(self):
        test_data = {}
        for line in self.test_data_reader:
            if line[0] == "PassengerId" : 
                test_data["Title"] = line
            else:
                PassengerId = int(line[0])
                test_data[PassengerId] = {}
                for index,title in enumerate(test_data["Title"]):
                    if index == 0 : continue
                    elif title == "Survived" : 
                        test_data[PassengerId][title] = int(line[index])
                    elif title == "Pclass" : 
                        test_data[PassengerId][title] = int(line[index])
                    elif title == "Sex":
                        if line[index] == "male":
                            test_data[PassengerId][title] = 1
                        else:
                            test_data[PassengerId][title] = -1
                    elif title == "Age":
                        test_data[PassengerId][title] = float(line[index]) if line[index] != "" else 0.0
                    elif title == "SibSp":
                        test_data[PassengerId][title] = int(line[index])
                    elif title == "Parch":
                        test_data[PassengerId][title] = int(line[index])
                    elif title == "Fare":
                        if line[index] != "":
                            test_data[PassengerId][title] = float(line[index])
                        else:
                            test_data[PassengerId][title] = 0
                    elif title == "Embarked":
                        if line[index] == "S":
                            test_data[PassengerId][title] = 1
                        if line[index] == "C":
                            test_data[PassengerId][title] = 2
                        if line[index] == "Q":
                            test_data[PassengerId][title] = 3
                    else:
                        test_data[PassengerId][title] = line[index]
        return(test_data)
    
    def Loading_Test_Lable(self,test_dict):
        for line in self.test_label_reader:
            if line[0] == "PassengerId" : continue
            PassengerId,Survived = line
            PassengerId = int(PassengerId)
            Survived = int(Survived)
            test_dict[PassengerId]["Survived"] = Survived
        return(test_dict)
    
    def Loading_Test_Data(self):
        test_dict = self.Loading_Test_Data_()
        test_dict = self.Loading_Test_Lable(test_dict)
        return(test_dict)
    
    def OutputMatrixData(self):
        train_data_matrix = []
        train_label_matrix = []
        test_data_matrix = []
        test_label_matrix = []
        
        for index in self.train_data_dict:
            if index == "Title":continue
            vector_buffer = [self.train_data_dict[index]["Pclass"],
                             self.train_data_dict[index]["Sex"],
                             self.train_data_dict[index]["Age"],
                             self.train_data_dict[index]["SibSp"],
                             self.train_data_dict[index]["Parch"],
                             self.train_data_dict[index]["Fare"],
                             self.train_data_dict[index]["Embarked"]]
            train_data_matrix.append(vector_buffer)
            train_label_matrix.append(self.train_data_dict[index]["Survived"])
            
        for index in self.test_data_dict:
            if index == "Title":continue
            vector_buffer = [self.test_data_dict[index]["Pclass"],
                             self.test_data_dict[index]["Sex"],
                             self.test_data_dict[index]["Age"],
                             self.test_data_dict[index]["SibSp"],
                             self.test_data_dict[index]["Parch"],
                             self.test_data_dict[index]["Fare"],
                             self.test_data_dict[index]["Embarked"]]
            test_data_matrix.append(vector_buffer)
            test_label_matrix.append(self.test_data_dict[index]["Survived"])
            
        train_data_matrix = np.array(train_data_matrix,dtype = "float32")
        train_label_matrix = np.array(train_label_matrix,dtype = "float32").transpose()
        test_data_matrix = np.array(test_data_matrix,dtype = "float32")
        test_label_matrix =  np.array(test_label_matrix,dtype = "float32").transpose()
        
        return(train_data_matrix,train_label_matrix,test_data_matrix,test_label_matrix)





if __name__ == "__main__":
    train_data_path = "C:/Yang/Project/MachineLearning/titanic/train.csv"
    test_data_path = "C:/Yang/Project/MachineLearning/titanic/test.csv"
    test_label_path = "C:/Yang/Project/MachineLearning/titanic/gender_submission.csv"
    
    titanic_data = TitanicData(   train_data_path=train_data_path,
                              test_data_path=test_data_path,
                              test_label_path=test_label_path)
    titanic_data.OutputMatrixData()
    
    
    
    
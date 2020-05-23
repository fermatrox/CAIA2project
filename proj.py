#
# proj.py
#

import os
import configparser
import random
import math
import numpy as np
import keras
from keras.preprocessing import image
import matplotlib.pyplot as plt
import logging
from datetime import datetime
import sys
import innvestigate
import innvestigate.utils as iutils
from warnings import simplefilter

simplefilter(action='ignore', category=FutureWarning)

class WorkSpace:

    def __init__(self):
        self.projectFolder = "F:/ud_MLEN/UU/CAIA_II/Project"
        self.wrkFolder = self.projectFolder + "/wrkT"      
        self.projectConfig = self.wrkFolder + "/project.conf"
        self.slices = 16
        self.dnn_model = None
        if self._file_exists(self.projectConfig):
            self.currentNr = self._read_config_value('PRORUN','currentnr')
            self._load_project_conf()
            self._load_model()
        else:
            self.currentNr = 0
            self._write_config_value('PRORUN','currentnr',self.currentNr)
            self._new_project()
            self._write_project_conf()
        
    def _status(self, status):
        self.status = status
        self._write_config_value(self.section,'status',self.status)
        
    def _load_project_conf(self):
        self.currentNr = self._read_config_value('PRORUN','currentnr')
        self.section = 'PRORUN#' + self.currentNr       
               
        self.datasetsize = int(self._read_config_value(self.section,'datasetsize'))
        self.trainingsize = int(self._read_config_value(self.section,'trainingsize'))
        self.validationsize = int(self._read_config_value(self.section,'validationsize'))
        self.remainingsize = int(self._read_config_value(self.section,'remainingsize'))
        self.testsize = int(self._read_config_value(self.section,'testsize'))       
        
        self.datafolder = self._read_config_value(self.section, 'datafolder')
        self.trainingfile = self._read_config_value(self.section,'trainingfile')
        self.validationfile = self._read_config_value(self.section,'validationfile')
        self.testfile = self._read_config_value(self.section,'testfile')
        self.remainingfile = self._read_config_value(self.section,'remainingfile')
        self.logfile = self._read_config_value(self.section,'logfile')
        self.modelfile = self._read_config_value(self.section,'modelfile')  
        
        self.datasetsizeprocent = int(self._read_config_value(self.section,'datasetsizeprocent'))
        self.trainingsizeprocent = int(self._read_config_value(self.section,'trainingsizeprocent'))
        self.testvalidationsplitprocent = int(self._read_config_value(self.section,'testvalidationsplitprocent'))
        
        self.model = int(self._read_config_value(self.section,'model'))
        self.epochs = int(self._read_config_value(self.section,'epochs'))
        self.batchsize = int(self._read_config_value(self.section,'batchsize'))
        self.enginebatchsize = int(self._read_config_value(self.section,'enginebatchsize'))
        self.enginemaxiter = int(self._read_config_value(self.section,'enginemaxiter'))
        self.verbose = int(self._read_config_value(self.section,'verbose'))
        self.splitmode = int(self._read_config_value(self.section,'splitmode'))
        self.framesize= int(self._read_config_value(self.section,'framesize'))       
        self._status("LOADED")
 
    def _write_project_conf(self):
        self._write_config_value('PRORUN','currentNr',self.currentNr)
        
        self._write_config_value(self.section,'datasetsize', self.datasetsize)
        self._write_config_value(self.section,'trainingsize', self.trainingsize)
        self._write_config_value(self.section,'validationsize', self.validationsize)
        self._write_config_value(self.section,'remainingsize', self.remainingsize)
        self._write_config_value(self.section,'testsize', self.testsize)
        
        self._write_config_value(self.section,'datafolder', self.datafolder)
        self._write_config_value(self.section,'trainingfile', self.trainingfile)
        self._write_config_value(self.section,'validationfile', self.validationfile)
        self._write_config_value(self.section,'testfile', self.testfile)
        self._write_config_value(self.section,'remainingfile', self.remainingfile)
        self._write_config_value(self.section,'logfile', self.logfile)
        self._write_config_value(self.section,'modelfile', self.modelfile)  
        
        self._write_config_value(self.section,'datasetsizeprocent', self.datasetsizeprocent)
        self._write_config_value(self.section,'trainingsizeprocent', self.trainingsizeprocent)
        self._write_config_value(self.section,'testvalidationsplitprocent', self.testvalidationsplitprocent)
        
        self._write_config_value(self.section,'model', self.model)
        self._write_config_value(self.section,'epochs', self.epochs)
        self._write_config_value(self.section,'batchsize', self.batchsize)
        self._write_config_value(self.section,'enginebatchsize', self.enginebatchsize)
        self._write_config_value(self.section,'enginemaxiter', self.enginemaxiter)
        self._write_config_value(self.section,'verbose', self.verbose)
        self._write_config_value(self.section,'splitmode', self.splitmode)
        self._write_config_value(self.section,'framesize', self.framesize)
 
    def _new_project(self, splitMode=2, datasetSizeProcent=100, trainingSizeProcent=70, validationSizeProcent=50):
        currentNr = 0
        config = configparser.ConfigParser()
        config.read(self.projectConfig)
        search = True
        while search:
            currentNr = currentNr + 1
            section = 'PRORUN#' + str(currentNr)
            if not section in config.sections():
                search = False
                self.currentNr = currentNr
                self.section = section
        self.prorunFolder = self.wrkFolder + "/" + self.section
        self._create_folder(self.prorunFolder) 
        self.datafolder = self.projectFolder + "/data/source100/slice"        
        self.trainingfile = self.prorunFolder + "/trainingFile"
        self.validationfile = self.prorunFolder + "/validationFile"
        self.testfile = self.prorunFolder + "/testFile"
        self.remainingfile = self.prorunFolder + "/remainingFile"
        self.logfile = self.prorunFolder + "/logFile"
        self.modelfile = self.prorunFolder + "/model.h5" 

        self.datasetsize = 0
        self.trainingsize = 0
        self.validationsize = 0
        self.remainingsize = 0
        self.testsize = 0
        
        self.datasetsizeprocent = datasetSizeProcent
        self.trainingsizeprocent = trainingSizeProcent
        self.testvalidationsplitprocent = validationSizeProcent
        
        self.model = 1
        self.epochs = 20
        self.batchsize = 100
        self.enginebatchsize = 200
        self.enginemaxiter = 50
        self.verbose = 1
        self.splitmode = splitMode
        self.framesize = 100  
        self._write_project_conf()
    
    def _generate_project_data(self):
        totalDataset = self._generate_dataset(self.datasetsizeprocent)
        trainingDataset = self._extract_subset(totalDataset, self.trainingsizeprocent)
        remainingDataset = self._list_subtraction(totalDataset, trainingDataset)
        validationDataset = self._extract_subset(remainingDataset, self.testvalidationsplitprocent)
        testDataset = self._list_subtraction(remainingDataset, validationDataset)   
        
        if self.splitmode == 2:
            trainingDataset = self._expand_subset(trainingDataset)
            remainingDataset = self._expand_subset(remainingDataset)
            validationDataset = self._expand_subset(validationDataset)
            testDataset = self._expand_subset(testDataset)
        
        self._write_list_to_file(trainingDataset,self.trainingfile)
        self._write_list_to_file(validationDataset,self.validationfile)
        self._write_list_to_file(testDataset,self.testfile)
        self._write_list_to_file(trainingDataset,self.remainingfile) 
        
        self.trainingsize = len(trainingDataset)
        self.remainingsize = len(remainingDataset)
        self.validationsize = len(validationDataset)
        self.testsize = len(testDataset)
        self.datasetsize = self.trainingsize + self.validationsize + self.testsize
        self._write_project_conf() 
        
    def _logg_write(self, message):
        now = datetime.now()
        current_time = now.strftime("%Y-%m-%d %H:%M:%S")
        with open(self.logfile, 'a') as f: 
            f.write(current_time + " " + message + "\n") 
        
    def _logg_info(self, message):
        #logging.info(message)
        self._logg_write("INFO: " + message)
        
    def _logg_warning(self, message):
        #logging.warning(message)
        self._logg_write("WARNING: " + message)
        
    def _logg_debug(self, message):
        #logging.debug(message)
        self._logg_write("DEBUG: " + message)
        
    def _logg_error(self, message):
        #logging.error(message)
        self._logg_write("ERROR: " + message)
    
    def _read_config_value(self, section, key):
        configValue = 0
        if self._folder_exists(self.projectConfig):
            config = configparser.ConfigParser()
            config.read(self.projectConfig)
            try:
                configValue = config[section][key]
            except:
                print("Key not found", key)
        return configValue  
        
    def _write_config_value(self, section, key, value):
        result = False
        try:
            config = configparser.ConfigParser()
            #load config file
            if self._file_exists(self.projectConfig):
                config.read(self.projectConfig)
            #create section if needed
            if not section in config.sections():
                config[section] = {}
            #add value to key
            config[section][key] = str(value)
            #save config file
            with open(self.projectConfig, 'w') as configfile:
                config.write(configfile)
                result = True
        except:
            print('config write error: failed to write to ', file)
        else:
            return result
        
    def _current_folder(self):
        return os.getcwd()
    
    def _change_current_folder(self, newFolder):
        os.chdir(newFolder)
    
    def _folder_content(self, folder):
        return [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]

    def _folders_in_folder(self, folder):
        return [f for f in os.listdir(folder) if os.path.isdir(os.path.join(folder, f))]

    def _folder_exists(self, folder):
        return os.path.exists(folder)

    def _file_exists(self, file):
        return os.path.isfile(file)

    def _create_folder(self, folder):
        if not os.path.exists(folder):
            os.makedirs(folder)
        
    def _remove_file(self, file):
        if self._file_exists(file):
            os.remove(file)
            
    def _write_list_to_file(self, dataList, file):
        with open(file, 'w') as filehandle:
            for listitem in dataList:
                filehandle.write('%s\n' % listitem)
            
    def _read_file_to_list(self, file):
        dataList = []
        with open(file, 'r') as filehandle:
            for line in filehandle:
                # remove linebreak which is the last character of the string
                currentItem = line[:-1]
                # add item to the list
                dataList.append(currentItem)
        return dataList

    def _list_subtraction(self, list1, list2):
        # list1 - list2, remove all elements in list2 from list1
        return [x for x in list1 if x not in list2]

    def _unique(self, notUniqueList):
        listSet = set(notUniqueList)
        unique_list = (list(listSet))
        return unique_list

    def _expand_subset(self, dataset):
        dataList = self._folders_in_folder(self.datafolder)
        result = []
        for i in dataset:
            for j in range(6):
                n = i + "_" + str(j+1)
                if n in dataList:
                    result.append(n)
        return result       

    def _extract_subset(self, dataset, extractionSize):
        if (extractionSize >= 0 and extractionSize <= 100):
            dataPoints = int(math.ceil(len(dataset) * (extractionSize / 100)))
            result = np.random.choice(dataset, dataPoints, replace=False)
        else:
            #out of bounds
            print("extractionSize is out of bounds [0-100]", extractionSize)
        return result

    def _generate_dataset(self, datasetSize):        
        if self.splitmode == 1:
            dataList = self._folders_in_folder(self.datafolder)
        if self.splitmode == 2:
            tmpList = []
            folderInFolder = self._folders_in_folder(self.datafolder)
            for i in folderInFolder:
                tmpList.append(i[:7])
            dataList = self._unique(tmpList)
        return self._extract_subset(dataList, datasetSize)
           
    def _save_model(self):
        self.dnn_model.save(self.modelfile)
        
    def _load_model(self):
        if self._file_exists(self.modelfile):
            self.dnn_model = keras.models.load_model(self.modelfile)

    def _remove_model(self):
        del self.dnn_model
        self.dnn_model = None

    def _load_datapoint(self, dataPointRef):
        content = self._folder_content(self.datafolder + "/" + dataPointRef)
        first = True
        for i in content:
            img = image.load_img(self.datafolder + "/" + dataPointRef + "/" + i)
            data = np.asarray(img)[:,:,0]
            if first:
                tensorData = [data]
                first = False
            else:
                tensorData.append(data)
        tensor = np.stack(tensorData, axis=2)
        return tensor

    def _get_target(self, dataPointRef):
        return dataPointRef[2:3]

    def _compare_code(self, a,b):
        if a == b:
            return 1
        else:
            return 0
        
    def _target_encoding(self, target):
        codeList = ['F','H','J','P','R','S','T','W']
        return np.asarray([self._compare_code(target,i) for i in codeList])

    def _create_dataset(self, DataPointsList):
        tensorList = []
        targetList = []
        for dataPoint in DataPointsList:
            tensorList.append(self._load_datapoint(dataPoint))
            targetList.append(self._target_encoding(self._get_target(dataPoint)))
        return np.stack(tensorList, axis=0), np.stack(targetList, axis=0)     

    def _fn_preprocessing(self, X):
        X /= 127.5       
        X+= -1
        return X
        
    def _fn_revert_preprocessing(self, X):
        X = X + 1
        X *= 127.5
        return X
        
    def _preprocess_functions(self):
        return (self._fn_preprocessing, self._fn_revert_preprocessing)

    def _preprocess(self, inputData, targetData):    
        xData = self._fn_preprocessing(inputData.astype("float32")) 
        yData = targetData
        return xData, yData

    def _create_model(self):
        self.testsize
        input_shape = (self.framesize, self.framesize, self.slices)
        if self.model == 1:
            self.dnn_model = keras.models.Sequential([
                keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=input_shape),
                keras.layers.Conv2D(64, (3, 3), activation="relu"),
                keras.layers.MaxPooling2D((2, 2)),
                keras.layers.Flatten(),
                keras.layers.Dense(512, activation="relu"),
                keras.layers.Dense(8, activation="softmax"),
                ])
        if self.model == 2:
            inputA = keras.layers.Input(shape=(4, self.framesize, self.framesize))
            inputB = keras.layers.Input(shape=(4, self.framesize, self.framesize))
            inputC = keras.layers.Input(shape=(4, self.framesize, self.framesize))
            inputD = keras.layers.Input(shape=(4, self.framesize, self.framesize))
            
            a = keras.layers.Conv2D(32, (3, 3), activation="relu") (inputA)
            a = keras.layers.BatchNormalization(axis=2) (a)
            a = keras.layers.MaxPooling2D((2, 2)) (a)
                    
            b = keras.layers.Conv2D(32, (3, 3), activation="relu") (inputB)
            b = keras.layers.BatchNormalization(axis=2) (b)
            b = keras.layers.MaxPooling2D((2, 2)) (b)
                    
            c = keras.layers.Conv2D(32, (3, 3), activation="relu") (inputC)
            c = keras.layers.BatchNormalization(axis=2) (c)
            c = keras.layers.MaxPooling2D((2, 2)) (c)
                    
            d = keras.layers.Conv2D(32, (3, 3), activation="relu") (inputD)
            d = keras.layers.BatchNormalization(axis=2) (d)
            d = keras.layers.MaxPooling2D((2, 2)) (d)
                    
            ab = keras.layers.concatenate( [a, b], axis=2)
            cd = keras.layers.concatenate( [c, d], axis=2)
            
            ab = keras.layers.Conv2D(20, (3, 3), activation="relu") (ab)
            ab = keras.layers.BatchNormalization(axis=2) (ab)
            ab = keras.layers.MaxPooling2D((2, 2)) (ab)
            
            cd = keras.layers.Conv2D(20, (3, 3), activation="relu") (cd)
            cd = keras.layers.BatchNormalization(axis=2) (cd)
            cd = keras.layers.MaxPooling2D((2, 2)) (cd)
            
            abcd = keras.layers.concatenate([ab, cd], axis=2)
            
            abcd = keras.layers.Conv2D(16, (3, 3), activation="relu") (abcd)
            abcd = keras.layers.Conv2D(64, (3, 3), activation="relu") (abcd)
            abcd = keras.layers.MaxPooling2D((2, 2)) (abcd)
            abcd = keras.layers.Dense(512, activation="relu") (abcd)
            abcd = keras.layers.Dense(8, activation="softmax") (abcd)
            self.dnn_model = keras.models.Model(inputs=[inputA, inputB, inputC, inputD], outputs=abcd)
        if self.dnn_model == None:
            print("Warning create_model failed")

    def _multiple_input_split(self, x):
        if self.model == 1:
            mix = [x]
        if self.model == 2:
            mix = [x[0:3,:,:], x[4:7,:,:], x[8:11,:,:], x[12:15,:,:]]
        return mix

    def _train_model(self, dataPointList):        
        (x,y) = self._create_dataset(dataPointList)
        (x1, y1) = self._preprocess(x, y)
        
        if self.dnn_model == None: #create model        
            self._create_model()
            
        multi_input = self._multiple_input_split(x1)
        
        #train model
        self.dnn_model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
        self.dnn_model.fit(multi_input, y1, epochs=self.epochs, batch_size=self.batchsize, verbose = self.verbose)
        
    def _evaluate_model(self, dataPointList):
        if self.dnn_model != None:
            (x,y) = self._create_dataset(dataPointList)
            (x1,y1) = self._preprocess(x,y)
            scores = self.dnn_model.evaluate(x1, y1, batch_size=self.batchsize)
            self._logg_info(f"Scores on dataset: loss={scores[0]} accuracy={scores[1]}")
            print(f"Scores on dataset: loss={scores[0]} accuracy={scores[1]}")
        else:
            print("Warning evaluate model failed")
      
    def _decode_prediction(self, pred):
        codeList = ['F','H','J','P','R','S','T','W']
        result = None
        p_score = 0
        for i in range(len(codeList)):
            if pred[i] > p_score:
                result = codeList[i]
                p_score = pred[i]
        return (result, p_score)

    def _get_categories(self):
        return {'F':1,'H':2,'J':3,'P':4,'R':5,'S':6,'T':7,'W':8}

    def _catNr(self, cat):
        return self._get_categories().get(cat)

    def _catCode(self, catNr):
        result = None
        dict_cat = self._get_categories()
        keys = dict_cat.keys()
        for k,v in dict_cat.items():
            if v == catNr:
                result = k
        return result
      
    #LRP
    def _get_img_from_tensor(self, tensor, slice):
        tensorList = [tensor[:,:,slice], tensor[:,:,slice], tensor[:,:,slice]]
        return np.stack(tensorList, axis=2)

    def _show_tensor(self, tensor, saveFile = ""):
        plt.subplots(4,4,figsize=(15,15))
        for i in range(16):
            plt.subplot(4,4,i+1)
            plt.imshow(self._get_img_from_tensor(tensor, i))
        if saveFile != "": 
            plt.savefig(saveFile, bbox_inches='tight')
        plt.show()
        
    def _get_hotspot_img(self, tensor, relevance, frameNr):
        t = self._get_img_from_tensor(tensor, frameNr)
        r = self._get_img_from_tensor(relevance, frameNr)
        r = r.sum(axis=np.argmax(np.asarray(r.shape) == 3))
        r /= np.max(np.abs(r))
        r = [r > 0.3]
        t[:,:,0][r] = 255
        return t
        
    def _get_relevance_img(self, relevanceTensor, frameNr):
        relevanceFrames = relevanceTensor.reshape(self.framesize,self.framesize,self.slices)
        img = self._get_img_from_tensor(relevanceFrames,frameNr)
        img = img.sum(axis=np.argmax(np.asarray(img.shape) == 3))  
        img /= np.max(np.abs(img))
        return img
        
    def _show_hotspot(self, tensor, relevance, saveFile = ""):
        plt.subplots(4,4,figsize=(15,15))
        for i in range(16):
            plt.subplot(4,4,i+1)
            t = self._get_hotspot_img(tensor, relevance, i)
            plt.imshow(t)
        if saveFile != "": 
            plt.savefig(saveFile, bbox_inches='tight')
        plt.show()
              
    def _test_result(self):
        testDataset = self._read_file_to_list(self.testfile)
        self._predict_test()
        calcTotal = len(testDataset)
        calcCorrect = 0
        calcWrong = 0
        result = []
        for i in range(len(testDataset)):
            cat, p = self._decode_prediction(self.p_score[i])
            catFacit = testDataset[i][2:3]
            result.append( (i, testDataset[i], catFacit ,cat ,p) )
            if cat == catFacit:
                calcCorrect = calcCorrect + 1
            else:
                calcWrong = calcWrong + 1
        return (result, calcCorrect, calcWrong, calcTotal)
    
    def _predict_test(self): 
        testDataset = self._read_file_to_list(self.testfile)
        (datasetX, datasetY) = self._create_dataset(testDataset)
        (datasetX, datasetY) = self._preprocess(datasetX, datasetY)
        self.p_score = self.dnn_model.predict(datasetX, verbose=1)
    
    def _prep_model(self):
        model = self.dnn_model
        model_wo_sm = iutils.keras.graph.model_wo_softmax(model)
        return model_wo_sm
        
    def _get_relevence_tensor(self, mowsm, prepdatasetX, tensorNr):
        prep_img_pack = prepdatasetX[tensorNr,:,:,:].reshape(1,self.framesize,self.framesize,self.slices)
        lrp_analyzer = innvestigate.analyzer.LRPZ(mowsm)
        relevanceTensor = lrp_analyzer.analyze(prep_img_pack).reshape(self.framesize,self.framesize,self.slices)
        return relevanceTensor
        
    def _get_relevance_per_frame(self, relevanceTensor):
        relPerFrame = []
        for i in range(self.slices):
            relevanceFrame = relevanceTensor[:,:,i]
            relPerFrame.append(np.sum(relevanceFrame))
        return relPerFrame
        
    def _get_hotspots_per_frame(self, relevanceTensor):
        treshold = 0.3
        a = relevanceTensor
        b = a / np.max(np.abs(a))
        c = [b > treshold]
        cinv = [b <= treshold]
        b[c] = 1
        b[cinv] = 0
        hotSpotPerFrame = []
        for i in range(self.slices):
            hotSpotFrame = b[:,:,i]
            hotSpotPerFrame.append(np.sum(hotSpotFrame))
        return hotSpotPerFrame
        
    #public methods
    
    def show_relevance_per_frame(self, mowsm, videoId, frameNr):
        dataFolderContent = self._folders_in_folder(self.datafolder)
        xaxis = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,14,15, 16]
        
        if videoId in dataFolderContent:
            (datasetX, datasetY) = self._create_dataset([videoId])
            (prepdatasetX, prepdatasetY) = self._preprocess(datasetX, datasetY)
            relevanceTensor = self._get_relevence_tensor(mowsm, prepdatasetX, 0)
            img_pack = datasetX[0]
            
            relPerFrame = self._get_relevance_per_frame(relevanceTensor)
            
            fig = plt.figure()
            ax1 = fig.add_subplot(111)
            ax1.bar(xaxis, relPerFrame, color='lightblue')
      
            ax1.set_xlim(0, 17)
            #ax1.set_ylim(-6.0, 10.0)
            ax1.set(title="Relevance distribution over frames", xlabel="FrameNr", ylabel="Relevance")
            plt.show()
            
        else:
            print("Error, Not in datafolder: ", videoId)
            
    def show_hotspot_per_frame(self, mowsm, videoId, frameNr):
        dataFolderContent = self._folders_in_folder(self.datafolder)
        xaxis = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,14,15, 16]
        
        if videoId in dataFolderContent:
            (datasetX, datasetY) = self._create_dataset([videoId])
            (prepdatasetX, prepdatasetY) = self._preprocess(datasetX, datasetY)
            relevanceTensor = self._get_relevence_tensor(mowsm, prepdatasetX, 0)
            img_pack = datasetX[0]
           
            hotSpotPerFrame = self._get_hotspots_per_frame(relevanceTensor)
            
            fig = plt.figure()
            ax1 = fig.add_subplot(111)
            ax1.bar(xaxis, hotSpotPerFrame, color='red')
      
            ax1.set_xlim(0, 17)
            ax1.set(title="Hotspot distribution over frames", xlabel="FrameNr", ylabel="Hotspot")
            plt.show()
            
        else:
            print("Error, Not in datafolder: ", videoId)
    
    def show_cmp_img_relevance(self, mowsm, videoId, frameNr):
        dataFolderContent = self._folders_in_folder(self.datafolder)
        if videoId in dataFolderContent:
            (datasetX, datasetY) = self._create_dataset([videoId])
            (prepdatasetX, prepdatasetY) = self._preprocess(datasetX, datasetY)
            relevanceTensor = self._get_relevence_tensor(mowsm, prepdatasetX, 0)
            img_pack = datasetX[0]
            
            relevance_img = self._get_relevance_img(relevanceTensor, frameNr)
            hotspot_img = self._get_hotspot_img(img_pack,relevanceTensor,frameNr)
            img = self._get_img_from_tensor(img_pack, frameNr)
            
            plt.subplots(1,3,figsize=(15,15))
            plt.subplot(1,3,1)
            plt.imshow(img)
            plt.subplot(1,3,2)
            plt.imshow(relevance_img, cmap="seismic", clim=(-1, 1))
            plt.subplot(1,3,3)
            plt.imshow(hotspot_img)
            plt.show()
        else:
            print("Error, Not in datafolder: ", videoId)
        
    def show_all_hotspot(self, mowsm, dataset):
        (datasetX, datasetY) = self._create_dataset(dataset)
        (prepdatasetX, prepdatasetY) = self._preprocess(datasetX, datasetY)
        for tensorNr in range(len(dataset)):
            savefile = self.wrkFolder + "/output/hspot" + dataset[tensorNr] + ".jpg"
            print(savefile)
            relevanceTensor = self._get_relevence_tensor(mowsm, prepdatasetX, tensorNr)        
            img_pack = datasetX[tensorNr]
            self._show_hotspot(img_pack, relevanceTensor, savefile)   
           
    def start_engine(self):
        runFile = self.wrkFolder + "/run." + self.currentNr
        stopFile = self.wrkFolder + "/stop." + self.currentNr
        self._remove_file(stopFile)
        self._write_list_to_file(['run'],runFile)
        
    def stop_engine(self):
        runFile = self.wrkFolder + "/run." + self.currentNr
        stopFile = self.wrkFolder + "/stop." + self.currentNr 
        self._write_list_to_file(['stop'],stopFile)
        
    def reset_training(self):
        runFile = self.wrkFolder + "/run." + self.currentNr
        stopFile = self.wrkFolder + "/stop." + self.currentNr   
        self.stop_engine()
        if self._file_exists(runFile):
            print("engine is still running")
        else:
            #copy trainingfile to remainingfile
            trainingDataset = self._read_file_to_list(self.trainingfile)
            self._write_list_to_file(trainingDataset,self.remainingfile)
            #remove model
            if self._file_exists(self.modelfile):
                self._remove_file(self.modelfile)
        
    def run_training(self):        
        counter = 0
        run = True 
        evalDataPointList = self._read_file_to_list(self.validationfile)       
        runFile = self.wrkFolder + "/run." + self.currentNr
        stopFile = self.wrkFolder + "/stop." + self.currentNr
        self._logg_info("Start training")
        
        self._load_model()
        self._status("TRAINING STARTED")
        while (run):    
            counter = counter + 1
            self._logg_info("Start Engine Batch: " + str(counter))
            remainingDataset = self._read_file_to_list(self.remainingfile)           
            if self._file_exists(stopFile):
                self._remove_file(runFile)
                self._status("TRAINING INTERUPTED")                
            if (self._file_exists(runFile)) and (counter < self.enginemaxiter) and (remainingDataset != []):
                #Execute next round    
                if (len(remainingDataset) > self.enginebatchsize):
                    batch = np.random.choice(remainingDataset, self.enginebatchsize, replace=False)
                    remainingDataset = self._list_subtraction(remainingDataset, batch)
                    #TRAIN MODEL HERE!
                    print("Counter: ", counter, len(remainingDataset))
                    self._train_model(batch)
                    self._evaluate_model(evalDataPointList)
                    self._write_list_to_file(remainingDataset, self.remainingfile)   
                    self._save_model()  
                else:
                    run = False
                    self._status("TRAINING COMPLETED")
            else:
                run = False
                self._status("TRAINING ENDED BY LIMIT")
                
        #evaluate model
        self._remove_model()
        self._load_model()
        self._logg_info("Final Model Evaluation:")
        self._evaluate_model(evalDataPointList)
        self._logg_info("Stop training")            
    
    def prorun_setup(self, splitMode, datasetSize, trainingSize, validationSize):
        self._new_project(splitMode, datasetSize, trainingSize, validationSize)
        self._generate_project_data()
        

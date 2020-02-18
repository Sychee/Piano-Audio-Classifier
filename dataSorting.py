import convFilter
import os, shutil
import matplotlib.pyplot as plt
from collections import defaultdict
import cv2
import split_folders

def countClasses(root, debugContent=False): #counts the classes and number of data from root
    class_count = 0
    data_count = 0
    for dataClass in os.listdir(root):
        class_count +=1
        for dataItem in os.listdir(root + "/" + dataClass):
            data_count += 1
        
        if debugContent:
            img = cv2.imread(root + "/" + dataClass + "/" + dataItem, cv2.IMREAD_GRAYSCALE)
            strImg = str(','.join(str(item) for innerlist in img for item in innerlist))
            print("DataClass: ", dataClass)
            print("Img String", strImg)
            
    print(f"Total Classes: {class_count}")
    print(f"Total Data: {data_count}")

def InitiateData(root): #clears all data in root folder
    for item in os.listdir(root):
        assert item in ["all", "train", "validation", "test", ""] #safety check to make sure root is correct
        shutil.rmtree(os.path.join(root,item)) #clears all items in root
    allDir = os.path.join(root, "all")
    os.mkdir(allDir)
    allallDir = os.path.join(allDir, "all")
    os.mkdir(allallDir) #creates all/all in root dir
    
def populateData(filtered_performances, root, save=True):
    """
    Slices spectrograms in filtered_performances and saves images in all/all folder with class as folder name
    """
    
    if save:
        InitiateData(root) #clear data in root folder
        
    root = os.path.join(root, "all/all")
    class_count = defaultdict(int)
    data_count = 0
    for piecenum in range(len(filtered_performances)):
        print(f"Piece {piecenum} of {len(filtered_performances)}")
        
        piece = filtered_performances[piecenum]
        try:
            performance = piece.load_performance(piece.available_performances[0], require_audio=False)
            spectrogram = performance.load_spectrogram()
        except Exception as e:
            print(f"EXCEPTION at Piece {piecenum}: {e}")
            continue
            
        slices = spectrogram.shape[1]
        for slice in range(slices):
            try:
                trueVal = str(int(''.join(map(str, convFilter.getNvec(slice, performance))), 2))
                if trueVal in class_count:
                    class_count[trueVal] += 1
                else:
                    class_count[trueVal] = 1
                data_count += 1
                
                trueSpec = convFilter.getSpectrogram(slice, performance)
                
                if save:
                    addImageToDirectory(trueSpec, f"img{class_count[trueVal]}.png", trueVal, root)
                
            except IndexError as e:
                print(f"INDEXERROR: PieceNum: {piecenum}, Slice: {slice}, Message: {e}")    
    
    print(f"Total Classes: {len(class_count)}")
    print(f"Total Data: {data_count}")
            
def addImageToDirectory(image, imageName, folder, root):
    """
    Adds image to directory specified
    """
    destDir = os.path.join(root, folder)
    if os.path.isdir(destDir):
        cv2.imwrite(os.path.join(destDir , imageName), image)
    else:
        try:  
            os.mkdir(destDir)  
            cv2.imwrite(os.path.join(destDir , imageName), image)
        except OSError as error:  
            print(error)

def divideDataIntoTrainValTestSets(root, train=.6, val=.2, test=.2):
    """
    Distributes data into train, validation, and test sets from all/all folder
    """
    allPath = os.path.join(root, "all/all")
    
    assert train + val + test == 1
    
    split_folders.ratio(allPath, output=root, seed=1337, ratio=(train, val, test)) # default values
    os.rename(os.path.join(root, "val"), os.path.join(root, "validation"))
    
    temp = "Temp"
    os.mkdir(os.path.join(root, "train" + temp))
    os.mkdir(os.path.join(root, "validation" + temp))
    os.mkdir(os.path.join(root, "test" + temp))
    
    for item in ["train", "validation", "test"]:
        dest = shutil.move(os.path.join(root, item), os.path.join(root, item + temp))
        os.rename(os.path.join(root, item + temp), os.path.join(root, item))

if __name__ == "__main__":
    DATA_ROOT_MSMD = '/Users/gbanuru/PycharmProjects/HACKUCI/msmd_aug_v1-1_no-audio/' # path to MSMD data set
    dataRoot = "/Users/gbanuru/PycharmProjects/HACKUCI/msmd/tutorials/data_root" # path to our created dataset

    filtered_performances = convFilter.filteredData(DATA_ROOT_MSMD) # creates a list with piece object
    populateData(filtered_performances[:50], dataRoot, save=True) # store data from 50 performances at data_root
    divideDataIntoTrainValTestSets(dataRoot) # divide data from data_root into train, validation, test sets
    countClasses(dataRoot + "/all/all") # count number of classes and data pieces
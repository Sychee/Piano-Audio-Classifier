#collect and split data into training, validation, and test sets

import convFilter
import os, shutil
import matplotlib.pyplot as plt
from collections import defaultdict
import cv2
import split_folders

def countClasses(data_root, debugContent=False): #counts the classes and number of data from root
    root = os.path.join(data_root, "all/all")
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
    
def initiateData(data_root): #CLEARS ALL DATA IN DATA_ROOT FOLDER
    for item in os.listdir(data_root):
        assert item in ["all", "train", "validation", "test", ""] #safety check to make sure root is correct
        shutil.rmtree(os.path.join(data_root,item)) #clears all items in root
    all_dir = os.path.join(data_root, "all")
    os.mkdir(all_dir)
    all_root = os.path.join(all_dir, "all")
    os.mkdir(all_root) #creates all/all in root dir
    
def populateData(filtered_performances, data_root, save=True):
    """
    Slices spectrograms in filtered_performances and saves images in all/all folder with class as folder name
    """
    if save:
        initiateData(data_root) #clear data in root folder
        
    all_root = os.path.join(data_root, "all/all")
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
            
        for slice in range(spectrogram.shape[1]):
            try:
                trueVal = str(int(''.join(map(str, convFilter.getNvec(slice, performance))), 2))
                trueSpec = convFilter.getSpectrogram(slice, performance)
                
                if trueVal in class_count:
                    class_count[trueVal] += 1
                else:
                    class_count[trueVal] = 1
                data_count += 1
                
                if save:
                    addImageToDirectory(trueSpec, f"img{class_count[trueVal]}.png", trueVal, all_root)
                
            except IndexError as e:
                print(f"INDEXERROR: PieceNum: {piecenum}, Slice: {slice}, Message: {e}")    
    
    print(f"Total Classes: {len(class_count)}")
    print(f"Total Data: {data_count}")
            
def addImageToDirectory(image, imageName, folder, root):
    """
    Adds image to directory specified
    """
    class_root = os.path.join(root, folder)
    if os.path.isdir(class_root):
        cv2.imwrite(os.path.join(class_root, imageName), image)
    else:
        try:  
            os.mkdir(class_root)  
            cv2.imwrite(os.path.join(class_root, imageName), image)
        except OSError as error:  
            print(error)

def divideDataIntoTrainValTestSets(data_root, train=.6, val=.2, test=.2):
    """
    Distributes data into train, validation, and test sets from all/all folder
    """
    all_root = os.path.join(data_root, "all/all")
    
    assert train + val + test == 1
    
    split_folders.ratio(all_root, output=data_root, seed=1337, ratio=(train, val, test)) # default values
    os.rename(os.path.join(data_root, "val"), os.path.join(data_root, "validation"))
    
    temp = "Temp"
    os.mkdir(os.path.join(data_root, "train" + temp))
    os.mkdir(os.path.join(data_root, "validation" + temp))
    os.mkdir(os.path.join(data_root, "test" + temp))
    
    for item in ["train", "validation", "test"]:
        dest = shutil.move(os.path.join(data_root, item), os.path.join(data_root, item + temp))
        os.rename(os.path.join(data_root, item + temp), os.path.join(data_root, item))
        
if __name__ == "__main__":
    DATA_ROOT_MSMD = '/Users/gbanuru/PycharmProjects/HACKUCI/msmd_aug_v1-1_no-audio/' # path to MSMD data set
    data_root = "/Users/gbanuru/PycharmProjects/HACKUCI/msmd/tutorials/data_root" # path to our created dataset  
    
    #filtered_performances = convFilter.filteredData(DATA_ROOT_MSMD) #creates a list with piece objects
    #print(f"All pieces: {len(filtered_performances)}")
    #populateData(filtered_performances[:], data_root, save = False)
    #divideDataIntoTrainValTestSets(data_root)
    
    countClasses(data_root)
    
    print("Done")
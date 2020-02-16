import convFilter
import os, shutil
import matplotlib.pyplot as plt
from collections import defaultdict
import cv2
import split_folders

def InitiateData(root): #clears all data in root folder and creates all/all folder
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
    InitiateData(root) #clear data in root folder
    root = os.path.join(root, "all/all")
    class_count = defaultdict(int)
    data_count = 0
    for piecenum in range(len(filtered_performances)):
        piece = filtered_performances[piecenum]
        performance = piece.load_performance(piece.available_performances[0], require_audio=False)
        spectrogram = performance.load_spectrogram()
        slices = spectrogram.shape[1]
        print(f"Piece {piecenum} of {len(filtered_performances)}")
        for slice in range(slices):
            try:
                trueVal = str(int(''.join(map(str, convFilter.getNvec(slice, performance))), 2))
                if trueVal in class_count:
                    class_count[trueVal] += 1
                else:
                    class_count[trueVal] = 1
                data_count += 1
                
                if save:
                    trueSpec = convFilter.getSpectrogram(slice, performance)
                    addImageToDirectory(trueSpec, f"img{class_count[trueVal]}.png", trueVal, root)
                
            except IndexError as e:
                print(f"IndexError: PieceNum: {piecenum}, Slice: {slice}, Message: {e}")    

    print(f"Classes: {len(class_count)}")
    print(f"Data Count: {data_count}")
    #print(f"All Classes: {sorted(class_count.items(), key=lambda k_v: k_v[1], reverse=True)}")
            
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
    # path to MSMD data set
    DATA_ROOT_MSMD = '/Users/gbanuru/PycharmProjects/HACKUCI/msmd_aug_v1-1_no-audio/'
    dataRoot = "/Users/gbanuru/PycharmProjects/HACKUCI/msmd/tutorials/data_root"

    filtered_performances = convFilter.filteredData(DATA_ROOT_MSMD) #creates a list with piece object
    populateData(filtered_performances[:5], dataRoot) #store data from 5 performances at data_root
    divideDataIntoTrainValTestSets(dataRoot) #divide data from data_root into train, validation, test sets
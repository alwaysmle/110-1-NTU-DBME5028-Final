# This is the final project of DBME 5028
## How to use
(1) clone the github repository (include all the .py file in the directory) <br>
(2) type sh download.sh to download model (best.pt)<br>
(3) put the training image folder and testing image folder in the main directory <br>
(4-a) execute train.py, it will output the model weight of each epoch <br> 
(4-b) execute inferece.py, it will output a file called result.csv  <br>

## Train model
train.py: <br>
#### first argument --data: input the folder of train and test <br>
Example : python "/data1/home/8B07/Anthony/simsiam/train.py" --data /data1/home/8B07/Anthony/simsiam/ 
## Export csv
Inference.py :  <br>
#### first argument --data: input the folder of train and test <br>
#### second argument --model <b> </b> : input the model's directory name <br>
Example : python "/data1/home/8B07/Anthony/simsiam/inference.py" --data /data1/home/8B07/Anthony/simsiam/ --model /data1/home/8B07/Anthony/simsiam/best.pt


## Package requirements:
No specific package is required in the project

## Code 
Our model is based on Simsiam and SimTriplet, https://github.com/hrlblab/SimTriplet
Im my_model.py, Line 17-41 75-161 will be similar to the https://github.com/hrlblab/SimTriplet/blob/main/models/simsiam.py

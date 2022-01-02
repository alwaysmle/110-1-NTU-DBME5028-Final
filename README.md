# This is the final project of DBME 5028
## Download model
please download best.pt model using download.sh and put in the same folder as train.py and inference.py
## Train model
train.py: <br>
Example : python "/data1/home/8B07/Anthony/simsiam/train.py" --data /data1/home/8B07/Anthony/simsiam/ 
#### first argument --data: input the folder of train and test <br>
## Export csv
Inference.py :  <br>
#### first argument --data: input the folder of train and test <br>
#### second argument --model <b> </b> : input the model's directory name <br>
Example : python "/data1/home/8B07/Anthony/simsiam/inference.py" --data /data1/home/8B07/Anthony/simsiam/ --model /data1/home/8B07/Anthony/simsiam/best.pt


## Package requirements:
requirements.txt is provided in the folder

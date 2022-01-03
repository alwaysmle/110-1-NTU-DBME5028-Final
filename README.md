# This is the final project of DBME 5028
## How to use
(1) clone the github repository (include all the .py file in the directory) <br>
(2) type sh download.sh to download model (best.pt)<br>
(3) execute train.py or inferece.py <br>

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
requirements.txt is provided in the folder

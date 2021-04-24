# DisP-V_TNNLS_Code

Disentangled Prototype Plus Variation Model (DisP+V).

This work was submitted to IEEE Transactions on Neural Networks and Learning Systems (TNNLS). In this package, we implement our DisP+V using Pytorch, and train/test the DisP+V model on CAS-PEAL (disguise) dataset.

The PEAL dataset can be downloaded in the link (https://drive.google.com/file/d/1OqAA81yUXbyRIh0c8EJFAnOFbmvBGNMH/view?usp=sharing).

The trained DisP+V model can be downloaded in the link (https://drive.google.com/file/d/1IliSX7Ma3D2F47P27eyz8Nlg0k_Q0I5u/view?usp=sharing).


--------------------------------------------------------------------------------
PEAL_ori_test: input face images 

PEAL_genpro_test: generated prototype images 

PEAL_genvar_test: generated variation images 

PEAL_genori_test: reconstructed face images 

PEAL_pro_test: true prototype images

------------------------------------------------------------------------
Train DisP+V model:

1. Open configPV_disguise.py 

set con.batch_size =16;

set conf.epochs = 1000;

set conf.file='./dataset/LoadPEAL200.txt';

set conf.nd=200;

set conf.TrainTag = True;

2. Open readerDisguise.py
3. 
set shuffle=True in def get_batch

3. Run TrainPV_disguise.py
4. 
the trained model will be saved in saved_modelDisguise

---------------------------------------------------------------------------
Face editing/interpolation

1. Open configPV_disguise.py 

set con.batch_size =1;

set conf.epochs = 1;

set conf.file='./dataset/LoadPEALA.txt';    % Face edit, the input face is a standard image

(set conf.file='./dataset/LoadPEALA1.txt';   % Face interpolation, the input face is a image containing variation)

set conf.file1='./dataset/LoadPEALB.txt';

set conf.TrainTag = False;

2. Open readerDisguise.py

set shuffle=False in def get_batch

set shuffle=True in def get_batch1

3. Run FaceEdit.py

choose a trained model (e.g., E340), and load it in def generateNewFace

All images are saved in PEAL_gennew_test folder

--------------------------------------------------------------------------
Generate prototype, variation, and reconstrued images

1. Open configPV_disguise.py 

set con.batch_size =1;

set conf.epochs = 1;

set conf.file='./dataset/LoadPEAL5.txt';

set conf.TrainTag = False;

2. Open readerDisguise.py

set shuffle=False in def get_batch

3. Run Gen_PV.py

choose a trained model (e.g., E340), and load it in def generateImg



The software is free for academic use, and shall not be used, rewritten, or adapted as the basis of a commercial product without first obtaining permission from the authors. The authors make no representations about the suitability of this software for any purpose. It is provided "as is" without express or implied warranty.

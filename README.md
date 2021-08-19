# DisPV_TNNLS_Code

Disentangled Prototype Plus Variation Model (DisP+V).

This work has been accepted by IEEE Transactions on Neural Networks and Learning Systems (TNNLS). In this package, we implement our DisP+V using Pytorch, and train/test the DisP+V model on CAS-PEAL (disguise) dataset.

The trained DisP+V model can be downloaded in the link (https://drive.google.com/file/d/1ALlTC23XwxJ9VH8nDJ22lvns5wWN3o3G/view?usp=sharing).


--------------------------------------------------------------------------------
PEAL_ori_test: input face images 

PEAL_genpro_test: generated prototype images 

PEAL_genvar_test: generated variation images 

PEAL_genori_test: reconstructed face images 

PEAL_pro_test: true prototype images for reference

------------------------------------------------------------------------
## Train DisP+V model:

### Step 1. Open configPV_disguise.py 

set con.batch_size =16;

set conf.epochs = 1000;

set conf.file='./dataset/LoadPEAL200.txt';

set conf.nd=200;

set conf.TrainTag = True;

### Step 2. Open readerDisguise.py

set shuffle=True in def get_batch

### Step 3. Run TrainPV_disguise.py

the trained model will be saved in saved_modelDisguise

--------------------------------------------------------------------------
## Generate prototype, variation, and reconstrued images

### Steo 1. Open configPV_disguise.py 

set con.batch_size =1;

set conf.epochs = 1;

set conf.file='./dataset/LoadPEAL5.txt';

set conf.TrainTag = False;

### Step 2. Open readerDisguise.py

set shuffle=False in def get_batch

### Step 3. Run Gen_PV.py

choose a trained model (e.g., E340), and load it in def generateImg

---------------------------------------------------------------------------
## Face editing/interpolation

### Step 1. Open configPV_disguise.py 

set con.batch_size =1;

set conf.epochs = 1;

set conf.file='./dataset/LoadPEALA.txt';    % Face edit, the input face is a standard image

(set conf.file='./dataset/LoadPEALA1.txt';   % Face interpolation, the input face is a image containing variation)

set conf.file1='./dataset/LoadPEALB.txt';

set conf.TrainTag = False;

### Step 2. Open readerDisguise.py

set shuffle=False in def get_batch

set shuffle=True in def get_batch1

### Step 3. Run FaceEdit.py

choose a trained model (e.g., E340), and load it in def generateNewFace





The software is free for academic use, and shall not be used, rewritten, or adapted as the basis of a commercial product without first obtaining permission from the authors. The authors make no representations about the suitability of this software for any purpose. It is provided "as is" without express or implied warranty.

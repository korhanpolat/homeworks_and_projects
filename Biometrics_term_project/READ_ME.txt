** I used Python 2.7 and packages that I used are listed in 'requierements.txt'.


*** dataset ***
download the images using this link into ./images folder
https://www.dropbox.com/sh/m0n0zkhrcchbdsr/AADidenXD1wxhGZ0WuwM-1u4a?dl=0


============================
*** runnning experiments ***
============================

Experiment ID refers to two different experiments:
	exp_id=4 --> 312 classes 1950 images (at least 4 img per class)
	exp_id=8 --> 64 classes 512 images (exactly 8 img per class)


*  If you want to perform the experiments on all 10 splits, you have to change the the range
   of for loops inside the functions below. 

*  If you want to try with the other experiment set, go and change exp_id = 8 in the 
   classification scripts. I did not add any argument parser.


Eigen-flukes    -->	Run 'classify_eigen.py' without any modification to get test set
experiment		accuracy for experiment 4, split 1.
			
			Each split runs about a minute.


SIFT experiment --> 	Run 'classify_sift.py' without any modification to get test set
			accuracy for experiment 4, split 1.
 
			Each split runs about a minute for exp_id = 8 and about 10 minutes
			for exp_id = 4. 




===================
***** scripts *****
===================

opencv_funcs	:	Contains many opencv functions that I used during my experiments.
			'first_n_good_matches()' function is used by SIFT detector.  

th_segmentation	:	I used this to experiment with thresholding methods. Did not
			work out well.

eigenfluke	:	Contains the functions such as PCA transforming, reconstruction,
			symmetry correction etc. that I used for eigenfluke classification. 

my_utils	:	Contains basic functions that are used for object storage and
			accuracy calvulations.

db_utils	:	Contains functions that I used when discarding some of the data
			and extracting file info from directory of images. Contains also 
			the function that I used for train test indicecs splitting. 




===================
***** folders *****
===================

/images 	-->	contains dataset

/obj		--> 	contains objects that are saved and used later on, such as 
			accuracy records as Pandas Dataframe

/obj/pca_obj	-->	PCA weights are saved here, in order to avoid computing PCA 
			if it already exists. This folder can grow very large. Therefore it is 
			set to not saving PCA in the current settings. If you want to use PCA for
			repeated experiments, enable it from 'PCA_Classification' initialization in
			'classify_eigen.py'

/obj/split_ind 	-->	contains train, validation and test split indices for two experiments.

/data_cleaning	-->	contains the scripts that I used for data cleaning and for other
			irrelevant preprocessing purposes. These scripts are not meant to be
			re-run, I only include them for reference.

=================
**** objects **** 
=================

train_info(exp_id) :	a Pandas DataFrame object that stores image path, Id and various 				
			other information 

uniq_ids(exp_id)   :	an object that stores unique ID names that belong to an 
			experiment set	





![Darknet Logo](http://pjreddie.com/media/files/darknet-black-small.png)

# Darknet #
Darknet is an open source neural network framework written in C and CUDA. It is fast, easy to install, and supports CPU and GPU computation.

For more information see the [Darknet project website](http://pjreddie.com/darknet).

For questions or issues please use the [Google Group](https://groups.google.com/forum/#!forum/darknet).

### modify for caculating recall
usage:

	./darknet

note: 

1. Pic folder path should be modified

2. weights file path should be modified

major modification:

1.modify darknet.c to relealize a simple function for detection

2.add function to read lines in a txt file (for reading Pos and Neg image in different folder)

usage detail:

1.fisrt put Pos and Neg images in different folders and generate txt files which contain full path name of Pos or Neg Pics

2.modify voc.data yolov3-voc.cfg to suit your own need

3.get weights file after training

4.modify several path name in darknet.c function -> mian()

some explanation for the parameters:

datasetPath_kevin: folder path (which contain Pos and Neg pic and txt file,txt files should be placed in the exact folder should not be placed insider some folder in datasetPath_kevin)

filename: useless but should be given something (a string for example)

.5 .5 : float thresh  float hier_thresh

0, 0: char *outfile(should be assigned 0 right in this time ,TO DO LIST in the future to realize saving pic after detection), int fullscreen


#include "darknet.h"
#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
extern void run_detector(int argc, char **argv);
extern void predict_main(int argc, char **argv);

int predict_kevin(network *net,char **names,image **alphabet,char *outfile,char *filename, float thresh, float hier_thresh){
  int nboxes = 0;int nboxes_higher_than_thresh=0;
  while(1){
    char buff[256];
    char *input = buff;
    float nms=.45;
    if(filename){
	strncpy(input, filename, 256);
    } else {
	printf("Enter Image Path: ");
	fflush(stdout);
	input = fgets(input, 256, stdin);
	if(!input) return 0;
	strtok(input, "\n");
    }
    image im = load_image_color(input,0,0);
    image sized = letterbox_image(im, net->w, net->h);
    layer l = net->layers[net->n-1];

    float *X = sized.data;
    double time=what_time_is_it_now();
    network_predict(net, X);
    //printf("%s: Predicted in %f seconds.\n", input, what_time_is_it_now()-time);
    
    detection *dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, 0, 1, &nboxes);
    //printf("%d\n", nboxes);
    //if (nms) do_nms_obj(boxes, probs, l.w*l.h*l.n, l.classes, nms);
    if (nms) do_nms_sort(dets, nboxes, l.classes, nms);
    draw_detections(im, dets, nboxes, thresh, names, alphabet, l.classes);
    //-------------------------------------
    //printf(" l.classes: %d\n", l.classes);
    {
      for(int i = 0; i < nboxes; ++i){
	  for(int j = 0; j < l.classes; ++j){
	      if (dets[i].prob[j] > thresh){
		nboxes_higher_than_thresh = nboxes_higher_than_thresh+1;
	      }
	  }
      }

    }
    //-------------------------------------
    free_detections(dets, nboxes);
    if(outfile){
	//save_image(im, outfile);
#ifdef OPENCV
	//make_window("predictions", 512, 512, 0);
	//show_image(im, "predictions", 0);
#endif
    }
    else{
	//save_image(im, "predictions");
#ifdef OPENCV
	//make_window("predictions", 512, 512, 0);
	//show_image(im, "predictions", 0);
#endif
    }
    
    
    free_image(im);
    free_image(sized);
    if (filename) break;
  }
  return nboxes_higher_than_thresh;
}

void test_detector_kevin(char *datacfg, char *cfgfile, char *weightfile, char *filename, float thresh, float hier_thresh, char *outfile, int fullscreen,char *datasetPath)
{
    //1. initial network
    list *options = read_data_cfg(datacfg);
    char *name_list = option_find_str(options, "names", "data/names.list");
    char **names = get_labels(name_list);
    
    //--------------------
    float tp,fp,tn,fn,ImageCount;
    tp=0;fp=0;tn=0;fn=0;ImageCount=0;
    char ValTxtPath[100],NegTxtPath[100],TestTxtPath[100],TrainTxtPath[100];
    int fakeVal,fakeTest,CountVal,CountTest;fakeVal=0;fakeTest=0;CountTest=0;CountVal=0;
    //--------------------
    image **alphabet = load_alphabet();
    network *net = load_network(cfgfile, weightfile, 0);
    set_batch_network(net, 1);
    srand(2222222);
    
    //2. detect and output detect result
    //int NumberOfExcavatorDetected= predict_kevin(net,names,alphabet,outfile,filename,thresh,hier_thresh);
    //printf("number of detection: %d\n",NumberOfExcavatorDetected);
    
    //3. caculate tp,fp,tn,fn
    strcpy(ValTxtPath,datasetPath);strcpy(NegTxtPath,datasetPath);strcpy(TestTxtPath,datasetPath);strcpy(TrainTxtPath,datasetPath);
    strcat(ValTxtPath,"Valnamelist.txt");strcat(NegTxtPath,"Negnamelist.txt");strcat(TestTxtPath,"Testnamelist.txt");strcat(TrainTxtPath,"Trainnamelist.txt");
    printf("TestTxtPath: %s\n",TestTxtPath);
    
    char szTest[1000] = {0};
    
    
    //TestTxtPath
    FILE *txtFile = fopen(TestTxtPath, "r");  
    if(NULL == txtFile)
    {  
        printf("failed to open dos.txt\n");  
        return ;  
    }
    while(!feof(txtFile))  
    {  
        memset(szTest, 0, sizeof(szTest));  
        fgets(szTest, sizeof(szTest) - 1, txtFile); // 包含了\n  
	szTest[strlen(szTest)-1]=0;
	if(strlen(szTest)==0 || strlen(szTest)==1) break;
        //printf("%s ", szTest); 
	int NumberOfExcavatorDetected= predict_kevin(net,names,alphabet,outfile,szTest,thresh,hier_thresh);
	CountTest++;
	//printf("number of detection: %d\n",NumberOfExcavatorDetected);
	if(NumberOfExcavatorDetected){tp++;}
	else {
	  fn++;
	  fakeTest++;
	  printf("wrong Test -> Neg: %s\n ", szTest); 
	  //printf("number of detection: %d\n",NumberOfExcavatorDetected);
	}
    }  
    fclose(txtFile);     
    printf("\nTest ok\n");
    /*
    //TrainTxtPath
    FILE *TraintxtFile = fopen(TrainTxtPath, "r");  
    if(NULL == TraintxtFile)
    {  
        printf("failed to open dos.txt\n");  
        return ;  
    }
    while(!feof(TraintxtFile))  
    {  
        memset(szTest, 0, sizeof(szTest));  
        fgets(szTest, sizeof(szTest) - 1, TraintxtFile); // 包含了\n  
	szTest[strlen(szTest)-1]=0;
	if(strlen(szTest)==0 || strlen(szTest)==1) break;
        printf("%s ", szTest); 
	int NumberOfExcavatorDetected= predict_kevin(net,names,alphabet,outfile,szTest,thresh,hier_thresh);
	printf("number of detection: %d\n",NumberOfExcavatorDetected);
	if(NumberOfExcavatorDetected){tp++;}
	else {
	  fn++;
	  printf("wrong Pos -> Neg: %s ", szTest); 
	}
    }  
    fclose(TraintxtFile);     
    printf("\nTrain ok\n");*/
    
    //ValTxtPath
    FILE *ValtxtFile = fopen(ValTxtPath, "r");  
    if(NULL == ValtxtFile)
    {  
        printf("failed to open pos.txt\n");  
        return ;  
    }
    while(!feof(ValtxtFile))  
    {  
        memset(szTest, 0, sizeof(szTest));  
        fgets(szTest, sizeof(szTest) - 1, ValtxtFile); // 包含了\n  
	szTest[strlen(szTest)-1]=0;
	if(strlen(szTest)==0 || strlen(szTest)==1) break;
        //printf("%s ", szTest); 
	int NumberOfExcavatorDetected= predict_kevin(net,names,alphabet,outfile,szTest,thresh,hier_thresh);
	CountVal++;
	//printf("Pos number of detection: %d\n",NumberOfExcavatorDetected);
	if(NumberOfExcavatorDetected){tp++;}
	else {
	  fn++;
	  fakeVal++;
	  printf("Val number of detection: %d\n",NumberOfExcavatorDetected);
	  //printf("wrong Val -> Neg: %s ", szTest); 
	}
    }  
    fclose(ValtxtFile);
    
    printf("\nPos ok\n");
    /*
    //NegTxtPath
    FILE *NegtxtFile = fopen(NegTxtPath, "r");  
    if(NULL == NegtxtFile)
    {  
        printf("failed to open neg.txt\n");  
        return ;  
    }
    while(!feof(NegtxtFile))  
    {  
        memset(szTest, 0, sizeof(szTest));  
        fgets(szTest, sizeof(szTest) - 1, NegtxtFile); // 包含了\n  
	szTest[strlen(szTest)-1]=0;
	if(strlen(szTest)==0 || strlen(szTest)==1) break;
        //printf("%s ", szTest); 
	int NumberOfExcavatorDetected= predict_kevin(net,names,alphabet,outfile,szTest,thresh,hier_thresh);
	printf("Neg number of detection: %d\n",NumberOfExcavatorDetected);
	if(NumberOfExcavatorDetected==0){tn++;}
	else {
	  fp++;
	  printf("wrong Neg -> Pos: %s ", szTest); 
	}
    }  
    fclose(NegtxtFile);     

    printf("\nNeg ok\n");*/
    
    printf("tp: %f , fn: %f \n",tp,fn);
    printf("tn: %f , fp: %f \n",tn,fp);
    printf("fakeTest: %d , fakeTestPercent: %.2f , fakeVal: %d , fakeValPercent: %.2f\n",fakeTest,((float)fakeTest/(float)CountTest), fakeVal,((float)fakeVal/(float)CountVal));
    float recall = tp/(tp+fn)*100;
    float accuracy = (tp+tn)/(tp+tn+fn+fp)*100;
    float precise = tp/(tp+fp)*100;
    printf("recall = %.2f%%  accuracy = %.2f%%  precise = %.2f%%\n",recall,accuracy,precise);
    
}

void setGPU(int argc, char **argv){
    gpu_index = find_int_arg(argc, argv, "-i", 0);
    if(find_arg(argc, argv, "-nogpu")) {
        gpu_index = -1;
    }

#ifndef GPU
    gpu_index = -1;
#else
    if(gpu_index >= 0){
        cuda_set_device(gpu_index);
    }
#endif
}

int  main(int argc, char **argv)
{
    //predict_main(argc,argv);
    setGPU(argc,argv);

    //run_detector(argc, argv);
    char *datacfg_kevin = "cfg/voc.data";
    char *cfgfile_kevin = "cfg/yolov3-voc.cfg";
    char *weightfile_kevin = "/home/kevin/workspace/20181015Excavator/0detect_result/result_weights/20181106_diffDays/yolov3-voc_final.weights";
    char *datasetPath_kevin = "/home/kevin/workspace/20181015Excavator/original_dataset/dataset_underUsing/";
    char *filename = "/home/kevin/workspace/20181015Excavator/original_dataset/dataset_underUsing/Pos/09_DJI_00161.jpg";
    char *outfile_kevin = "/home/kevin/workspace/20181015Excavator/original_dataset/dataset_underUsing/09_DJI_00161.jpg";
    test_detector_kevin(datacfg_kevin, cfgfile_kevin, weightfile_kevin, filename, .5, .5, 0, 0,datasetPath_kevin);
    return 0;
}
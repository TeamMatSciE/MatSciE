import statistics as stats
import math

def rmsValues(path):
    out_fp = open(path + '/RMS_Vals.txt', 'w')
    for i in range(1,6):
        fp = open(path + "/fold" + str(i) + ".txt", 'r')
        rms_val = 0.0
        
        for j,line in enumerate(fp.readlines()):
            #print(j)
            if(j>4):
                break
            nums = [float(num) for num in (line[:-1].split("\t"))]
            rms_val += (100.0-nums[0])**2
        
        #macro_avg = [str(round(macro_avg[i]/5,2)) for i in range(3)]
        rms_val = math.sqrt(rms_val/5.0)
        #print('\t'.join(macro_avg))
        fp.close()
        #fp = open(path + "/fold" + str(i) + ".txt", 'a')
        #fp.write('\t'.join(macro_avg))
        #fp.close()
        out_fp.write("{:.2f}".format(rms_val) + '\n')

    out_fp.close()
    print("\n\nDone!")
    
    
def rmsValues_ovr(file_path):
    out_fp = open(file_path + '_RMS_Ovr.txt', 'w')
    fp = open(file_path + '.txt', 'r')
    rms_val = 0.0
    
    for j,line in enumerate(fp.readlines()):
        if(j>4):
            break
        nums = [float(num) for num in (line[:-1].split(","))]
        rms_val += (100.0-nums[0])**2
    
    rms_val = math.sqrt(rms_val/5.0)
    out_fp.write("{:.2f}".format(rms_val) + '\n')
    
    out_fp.close()
    fp.close()
    print("\n\nDone!")



def computeMacroAverage(path):
    for i in range(2,6):
        fp = open(path + "/fold" + str(i) + ".txt", 'r')
        macro_avg = [0.0, 0.0, 0.0]
        for line in fp.readlines():
            nums = [float(num) for num in (line[:-1].split("\t"))]
            macro_avg = [(macro_avg[i] + nums[i]) for i in range(3)]
        
        macro_avg = [str(round(macro_avg[i]/5,2)) for i in range(3)]
        #print('\t'.join(macro_avg))
        fp.close()
        fp = open(path + "/fold" + str(i) + ".txt", 'a')
        fp.write('\t'.join(macro_avg))
        fp.close()

    print("\n\nDone!")    


#n = sample size, n = 5 in our case
def CI(stdev, n):
    #the following is for alpha=0.025 and df = 4
    #t-value = 2.776
    return (round(2.776*stdev/math.sqrt(n), 2))
    


def computeStats(data, path):
    #data[fold][tag][P,R,F1]
    #results_fp = open('Computed_Stats.txt', 'w')
    results_fp = open(path+'.txt', 'w')
    
    vals = []
    #6 times =  5 tags + macro score
    for i in range(6):
        vals_p = []
        vals_r = []
        vals_f1 = []
        #5 times = 5 folds
        for j in range(5):
            vals_p.append(data[j][i][0])
            vals_r.append(data[j][i][1])
            vals_f1.append(data[j][i][2])
            
        mean_p = round(stats.mean(vals_p),2)
        stdev_p = round(stats.stdev(vals_p),2)
        ci_p = CI(stdev_p, 5)
        #print(ci_p)
        
        mean_r = round(stats.mean(vals_r),2)
        stdev_r = round(stats.stdev(vals_r),2)
        ci_r = CI(stdev_r, 5)
        #print(ci_r)
            
        mean_f1 = round(stats.mean(vals_f1),2)
        stdev_f1 = round(stats.stdev(vals_f1),2)
        ci_f1 = CI(stdev_f1, 5)
        #print(ci_f1)
        
        res = str(mean_p) + ',' + str(stdev_p) + ',' + str(ci_p) +'\t' + str(mean_r) + ',' + str(stdev_r) + ',' + str(ci_r) + '\t' + str(mean_f1) + ',' + str(stdev_f1) + ',' + str(ci_f1)
        results_fp.write(res + '\n')
    
    results_fp.close()    
        

#Computes the mean, standard deviation and confidence Interval for Precision, Recall and F1-Scores 
# of each tag, using the 5-fold data
def computeConfidenceIntervalsV2(path):
    #Runs the loop 5 times (for 5 folds)  
    fold = []   #will contain the data for all 5 folds
    for i in range(1,6):
        #open file , fold<i>.txt
        fp = open(path + "/fold" + str(i) + ".txt", 'r')
        fold_temp = []
        for line in fp.readlines():
            nums = [float(num) for num in (line[:-1].split("\t"))]
            #print(nums)
            fold_temp.append(nums)
        
        #print(fold_temp)
        fold.append(fold_temp)
        #print("\n\n")
        fp.close()
    
    computeStats(fold, path)
    #print("\n\nDone!")
    
    #pass fold[0][0], fold[1][0], fold[2][0], fold[3][0], fold[4][0], 
    print("Done")


def main():
    #computeMacroAverage("Baseline")
    '''
    computeConfidenceIntervalsV2("Bert_Domain")
    computeConfidenceIntervalsV2("Mimicking_Model")
    computeConfidenceIntervalsV2("Naive_Method")
    computeConfidenceIntervalsV2("perlines_Positive")
    computeConfidenceIntervalsV2("persection_all_lines")
    computeConfidenceIntervalsV2("Relabeling_Model")
    computeConfidenceIntervalsV2("Baseline")
    '''
    
    
    
    rmsValues_ovr("Bert_Domain")
    rmsValues_ovr("Mimicking_Model")
    rmsValues_ovr("Naive_Method")
    rmsValues_ovr("perlines_Positive")
    rmsValues_ovr("persection_all_lines")
    rmsValues_ovr("Relabeling_Model")
    rmsValues_ovr("Baseline")
    
    #rmsValues("Relabeling_Model")
    
    
main()


# coding: utf-8

# In[ ]:


'my student Id is 1220606621'
import numpy
import scipy.io
import math
import geneNewData

def mean_var_feature(file,feature):
        if feature=='mean':
            mean_array=numpy.array([numpy.mean(i) for i in file])
            mean=numpy.mean(mean_array)
            var=numpy.var(mean_array,ddof=1)
            return mean,var
        elif feature=='var':
            var_array=numpy.array([numpy.var(i) for i in file])
            mean=numpy.mean(var_array)
            var=numpy.var(var_array,ddof=1)
            return mean,var
        else:
            print('the given feature is wrong')
def normal_distribution_probability(x,mean,var):
    pi=math.pi
    return (1/((2*pi*var)**.5))*math.exp(-.5*(((x-mean)**2)/var)) 
def bayes_parameters(train_file,test_file):
    mean_mean,var_mean=mean_var_feature(train_file,'mean')
    mean_var,var_var=mean_var_feature(train_file,'var')
    mean_test_samples=numpy.array([numpy.mean(i) for i in test_file])
    var_test_samples=numpy.array([numpy.var(i) for i in test_file])
    var_bayes=numpy.vectorize(normal_distribution_probability)(var_test_samples,mean_var,var_var)
    mean_bayes=numpy.vectorize(normal_distribution_probability)(mean_test_samples,mean_mean,var_mean)
    
    return var_bayes*mean_bayes
def prediction(digit,train0,test0,train1,test1):
    
    
    if digit=='0':
        digit0_bayes_0=bayes_parameters(train0,test0)
        digit0_bayes_1=bayes_parameters(train1,test0)
        unconditional_prob=digit0_bayes_0+digit0_bayes_1
        prob_0=digit0_bayes_0*.5/unconditional_prob
        prob_1=digit0_bayes_1*.5/unconditional_prob
        
        prob=numpy.array(prob_0>prob_1,dtype=numpy.float64)
        
        return (sum(prob)/len(prob))
    else:
        digit1_bayes_1=bayes_parameters(train1,test1)
        digit1_bayes_0=bayes_parameters(train0,test1)
        unconditional_prob=digit1_bayes_1+digit1_bayes_0
        prob_0=(digit1_bayes_0*.5)/unconditional_prob
        prob_1=(digit1_bayes_1*.5)/unconditional_prob
        
        prob=numpy.array(prob_1>prob_0,dtype=numpy.float64)
        
        return sum(prob)/len(prob)
        
def main():
    myID='6621'
    geneNewData.geneData(myID)
    Numpyfile0 = scipy.io.loadmat('digit0_stu_train'+myID+'.mat')
    Numpyfile1 = scipy.io.loadmat('digit1_stu_train'+myID+'.mat')
    Numpyfile2 = scipy.io.loadmat('digit0_testset'+'.mat')
    Numpyfile3 = scipy.io.loadmat('digit1_testset'+'.mat')
    train0 = Numpyfile0.get('target_img')
    train1 = Numpyfile1.get('target_img')
    test0 = Numpyfile2.get('target_img')
    test1 = Numpyfile3.get('target_img')
    print([len(train0),len(train1),len(test0),len(test1)])
    print('Your trainset and testset are generated successfully!')
    print(mean_var_feature(train0,'mean'))
    print(mean_var_feature(train0,'var'))
    print(mean_var_feature(train1,'mean'))
    print(mean_var_feature(train1,'var'))
    print(prediction('0',train0,test0,train1,test1))
    print(prediction('1',train0,test0,train1,test1))


if __name__ == '__main__':
    main()


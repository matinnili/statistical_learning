


from Precode2 import *
import numpy
data = np.load('AllSamples.npy')
import matplotlib.pyplot as plt
import matplotlib




k1,i_point1,k2,i_point2 = initial_S2('6621') # please replace 0111 with your last four digit of your ID




print(k1)
print(i_point1)
print(k2)
print(i_point2)





def initial(point,k,data):
    center=point.reshape(1,2)
    
    for j in range(k-1):
        min_arg=0
        distance=np.mean(np.array([(data[:,0]-center[i,0])**2+(data[:,1]-center[i,1])**2 for i in range(len(center))]).T,1)
        print(distance.shape)
        min_arg=int(np.argmax(distance,0))    
        print(min_arg)
        center=np.concatenate([center,data[min_arg,:].reshape(1,2)],0)
        data=np.delete(data,min_arg,0)
    return center





def assigncluster(data,center):
    i_point1=center
    distance=np.array([(data[:,0]-i_point1[i,0])**2+(data[:,1]-i_point1[i,1])**2 for i in range(len(i_point1))]).T
    clusters=np.argmin(distance,1).reshape(300,1)
    output=np.concatenate([data,clusters],axis=1)
    return output
    




def loss(aug_matrix,center):
    sum_dis=0
    for i in range(len(center)):
        matrix_i=aug_matrix[aug_matrix[:,2]==i]
        distance=np.sum(np.array((matrix_i[:,0]-center[i,0])**2+(matrix_i[:,1]-center[i,1])**2))
        sum_dis+=distance
    return sum_dis





def centers(aug_matrix,k):
    centers=np.zeros([k,2])
    for i in range(k):
        centers[i]=np.mean(aug_matrix[aug_matrix[:,2]==i],0)[:2]
    return centers





center=initial(i_point1,4,data)
for i in range(20):
    output=assigncluster(data,center)
    loss_1=loss(output,center)
    print(loss_1)
    center=centers(output,4)
    print(center)

Loss_=[]
for j in range(2,10):
    center=initial(i_point2,j,data)
    for i in range(20):
        output=assigncluster(data,center)
        loss_1=loss(output,center)
        print(loss_1)
        center=centers(output,j)
        print(center)
    Loss_.append(loss_1)
fig = plt.figure()
plt.plot(range(2,10),Loss_[:])
ax=plt.axes()
ax.set_xlabel('number_of_cluster', fontsize=18)
ax.set_ylabel('Loss', fontsize=16)    
ax.set_title('Loss vs. number_of_clusters in second strategy',fontsize=15)
colors = ['red','green','blue','purple','orange','yellow']
plt.scatter(output[:,0],output[:,1], c=output[:,2], cmap=matplotlib.colors.ListedColormap(colors))
ax=plt.axes()
ax.set_xlim(0,10)
ax.set_ylim(0,10)
ax.set_title('6 cluster assignment in 2nd strategy')
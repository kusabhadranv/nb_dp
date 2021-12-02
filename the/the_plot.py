import matplotlib.pyplot as plt
import os
import numpy as np

encoding = ["DE","SUE","OUE","SHE","THE"]
ytick = [0,10,20,30,40,50,60,70,80,90,100]


filename = ["car","chess","mushroom","connect","nursery"] 
title = ["Car_evaluation","Chess","Mushroom","Connect-4","Nursery"]
accuracy = [96.3,95.6,98.2,75.7,90.4,91.2]
color=['green','blue','orange','red','black']
for ind,file in enumerate(filename):
	filepath = os.path.join(os.path.dirname(__file__),file+"_THE.csv")
	f = open(filepath)
	res_mat = np.loadtxt(f,delimiter=',')
	f.close() 

	plt.plot(res_mat[1:,0],res_mat[1:,1],label=title[ind],marker="o",c=color[ind])
	# plt.axhline(y=accuracy[ind],linestyle='--',c=color[ind])	
	

plt.yticks(ytick)
plt.xticks([1,2,3,4,5])

plt.title('THE Encoding')
plt.xlabel('Epsilon')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.savefig("THE.png")
plt.show()



# resFileName = os.path.join(os.path.dirname(__file__), 'results\\'+dNames[dtID]+'_encMethod_'+enc[encID]+'.csv')








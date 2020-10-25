import pandas as pd
import numpy as np
import random as rd
from sklearn.decomposition import PCA
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

class Pca_kmean:

    def __init__(self):
        pass

    def dataProcess(self):
        #tmm = pd.read_csv("TMMnorm2.csv", index_col = [*'Melanoma', *'COLON', *'BREAST'])

        tmm = pd.read_csv("TMMnorm4.csv", index_col = 'GeneID')

        #To return the index column as a list
        index_list = tmm.index.tolist()

        #Find the average value for each numeric column

        #Before do PCA, have to center and scale data.
        #after centering average value for each gene will be 0
        #After scaling, the standard deviation for the values for each gene will be 1.

        #Notice taht we are passing in the transpose of our data.  The scale function expect the sample to
        #be in rows instead of columns

        #We use samples as columns in this example because that is oftern how genomic data is stored.
        #If you have other data, you can store it however is easiest for you.  Just be aware if samples
        #are columns it will have to be transformed before analysis

        scaled_data = preprocessing.scale(tmm.T)
     
        kmeans_model = KMeans(n_clusters = 3, random_state=1)
        kmeans_model.fit(scaled_data)
        labels = kmeans_model.labels_
        print(labels)
        
        #Now create PCA object. Rather than just have a function that does PCA and return results, sklearn uses
        #objects that can be trained using one dataset and applied to another dataset.
                                      
        pca = PCA(2)

        #then we call the fit method. This is where we do all of the PCA math (i.e. calculate loading scores
        #and the variation each principle component accounts for

        pca.fit(scaled_data)

        #And this is where we generate coordinates for a PCA graph based on the loading scores and the scaled
        #data.

        pca_data = pca.transform(scaled_data)
        plt.scatter(x=pca_data[:,0], y = pca_data[:,1], c = labels)
        plt.show()
        #__________________________________________________________
        #Drawing Graph
        #start with Scree plot to see how many components should go into the final plot
        #1--First thing that we do is calculate the percentage of variation that each
        #principle component accounts for


        per_var = np.round(pca.explained_variance_ratio_*100, decimals=1)

        #2---Now, we create labels for the scree plot.  These are "PC1", "PC2", etc.
        #(one label peer principle component).

        labels = ['PC' + str(x) for x in range (1,len(per_var) +1)]

        #Now we use matplotlib to create a barplot

        plt.bar(x = range(1, len(per_var) +1), height = per_var, tick_label = labels)

        plt.ylabel("Percentage of Explained variance")
        plt.xlabel("Principle Component")
        plt.title("Scree Plot")
        plt.show()


        #To draw a PCA plot, we'll first put the new coordinates, created by
        #pca-transform (scaled.data), into a nice matrix where the rows have sample
        #labels and the columns have PC labels

        pca_df = pd.DataFrame(pca_data, index =['Melanoma1', 'Melanoma3', 'Melanoma4', 'Melanoma5',
                                        'COLON1', 'COLON2','COLON4','COLON5', 'BREAST1', 'BREAST2', 'BREAST3', 'BREAST4'], columns = labels)

        #These commands draw a scatterplot with a title and nice axis labels

        plt.scatter(pca_df.PC1, pca_df.PC2)
        plt.title("My PCA_Graph")
        plt.xlabel("PC1-{0}%".format(per_var[0]))
        plt.ylabel("PC2 - {0}%".format(per_var[1]))

        #Loop to add Sample names to graph

        for sample in pca_df.index:
            plt.annotate(sample, (pca_df.PC1.loc[sample],pca_df.PC2.loc[sample]))
        plt.plot(range(1, 100))
        scale_factor = 1

        xmin, xmax = plt.xlim()
        ymin, ymax = plt.ylim()

        plt.xlim(xmin * scale_factor, xmax * scale_factor)
        plt.ylim(ymin * scale_factor, ymax * scale_factor)

        #Display Graph
        plt.show()

        #Lastly, Let's look at the loadiing scores for PCq to determine which genes
        #had the largest influence on separating the two clusters along the x-axis

        #We start by creating a pandas "series object with the loading scores in PC1
        #Note: the PC's are zero-indexed so PC1 =0

        loading_scores = pd.Series(pca.components_[0], index = index_list)

        #Now we sort the loading scores based on their magnitude (absolute value)

        sorted_loading_scores = loading_scores.abs().sort_values(ascending = False)

        #Here get the names of the top 10 indexes(which are the gene names)

        top_10_genes = sorted_loading_scores[0:100].index.values
        print (type(top_10_genes))
        #print out Top 10 gene names and their corresponding loading scores

        print (loading_scores[top_10_genes])
        print ("_________________________________________________")
        pca_1 = []
        for num in top_10_genes:
           pca_1.append(num)


        print ("__________________________________________________________")
        


        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

        loading_scores = pd.Series(pca.components_[1], index = index_list)

        sorted_loading_scores = loading_scores.abs().sort_values(ascending = False)

        top_10_genes = sorted_loading_scores[0:100].index.values
        print (loading_scores[top_10_genes])
        pca_2 = []
        for num in top_10_genes:
            pca_2.append(num)
           

        print ("_______________________________________________________________")







def main():
    pk = Pca_kmean()
    pk.dataProcess()
    

main()
    

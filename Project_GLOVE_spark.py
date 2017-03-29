from pyspark.sql.types import *
from scipy import sparse
import numpy as np
from pyspark.mllib.linalg import Vectors, DenseMatrix
from pyspark.mllib.linalg.distributed import IndexedRow, IndexedRowMatrix, RowMatrix
from pyspark.mllib.linalg import Matrices
from pyspark.mllib.linalg.distributed import BlockMatrix
from math import log
import random
import re
from string import digits
from sklearn.metrics.pairwise import cosine_similarity
from pyspark.mllib.linalg import Vectors, VectorUDT
from pyspark.sql.types import ArrayType, FloatType
from pyspark.sql.functions import udf
# file path /FileStore/tables/lsozbvys1486565070283/corpus_test_glove_project.txt

################################# FUNCTION ##########################################################
#Funtion to extract word 

def removePunctuation(text):
    text=re.sub('[^A-Za-z0-9]+', ' ', text.lower())
    text = re.sub("\d+", " ", text)
    text=re.sub("\s\s+" , " ", text)
    return text
    #return re.sub(r'^\s+|\s+$|[^A-Za-z\d\s]', ' ', text.lower())

def remove_space_and_extract_word(line):
  line = line.replace(",", " ")
  line = line.split(" ")
  #line=list(set(line))
  #line=[str(x) for x in line]
  return line

#Transform se
def sentences_to_token_ids(line,id2word_braodcast):
    tokens=line.strip().split()
    token_ids = [id2word_braodcast.value[word][0] for word in tokens]
    return token_ids
  
#Return only value where key is in mincount word list id.
def check_value(cell,id2word_mincount):
  if(cell[0][0] in id2word_mincount and cell[0][1] in id2word_mincount):
    return (cell[0][0],cell[0][1],cell[1])
  
def build_data(j,vector_size):
    import random
    vector=(np.random.rand(vector_size) - 0.5) / float(vector_size + 1)
    vector=vector.tolist()
    biases=random.uniform(-0.5,0.5)
    gradient_squared=np.ones((vector_size),dtype=np.float64)
    gradient_squared=gradient_squared.tolist()
    gradient_squared_biaises=random.uniform(-0.5,0.5)
    return (j,vector,biases,gradient_squared,gradient_squared_biaises)

#Compute the coocurence matrix 
def compute_coocurence(token_ids):
  coocurences=[]
  window_size=10
  #for center_id in token_ids_list:
  # Collect all word IDs in left window of center word
  for center_i, center_id in enumerate(token_ids):
    context_ids = token_ids[max(0, center_i - window_size) : center_i]
    contexts_len = len(context_ids)
    for left_i, left_id in enumerate(context_ids):
      # Distance from center word
      distance = contexts_len - left_i
      # Weight by inverse of distance between words
      increment = 1.0 / float(distance)
      values_1=((center_id,left_id),increment)
      values_2=((left_id,center_id),increment)
      coocurences.append(values_1)
      coocurences.append(values_2)
  return coocurences


def build_W_final(id2word,W):
  word2id=id2word.map(lambda x: (x[1][0],x[0]))
  word2id_broadcast=sc.broadcast(word2id.collectAsMap())
  W_final={}
  for i in range(vocab_size):
    word=word2id_broadcast.value[i]
    if i in id2word_mincount:
      W_final[word]=W.value[i]
    else:
      W_final[word]=(np.random.rand(vector_size) - 0.5) / float(vector_size + 1)
  return W_final


#Find Top similar 
def similarity(W_final,word1,n):
  most_similar_words = {}
  main_word = W_final['word1']
  for i in W_final:
      cosine_sim = cosine_similarity(main_word, W_final[i])
      result=(i,cosine_sim)
      most_similar_words[i]=cosine_sim
  sorted_x = sorted(most_similar_words.items(), key=operator.itemgetter(1),reverse=True)
  return sorted_x[:n]

def compute_error(x,learning_rate,x_max,alpha):
  
        cooccurrence=x[1][8]
    
        main_vector=np.asarray(x[1][0])
        biase_main=x[1][1]
        gradient_squared_main=np.asarray(x[1][2])
        gradient_squared_biases_main=x[1][3]
        
        context_vector=np.asarray(x[1][4])
        biase_context=x[1][5]
        gradient_squared_context=np.asarray(x[1][6])
        gradient_squared_biases_context=x[1][7]
        
        
        weight = (cooccurrence / x_max) ** alpha if cooccurrence < x_max else 1
        # Compute inner component of cost function, which is used in
        # both overall cost calculation and in gradient calculation
        #
        #   $$ J' = w_i^Tw_j + b_i + b_j - log(X_{ij}) $$
        cost_inner = (main_vector.dot(context_vector)
                      + biase_main + biase_context
                      - log(cooccurrence))
        # Compute cost
        #
        #   $$ J = f(X_{ij}) (J')^2 $$
        cost = weight * (cost_inner ** 2)

        # Add weighted cost to the global cost tracker

        # Compute gradients for word vector terms.
        #
        # NB: `main_word` is only a view into `W` (not a copy), so our
        # modifications here will affect the global weight matrix;
        # likewise for context_word, biases, etc.
        grad_main = learning_rate*weight * cost_inner * main_vector
        # Compute gradients for bias terms
        grad_bias_main = learning_rate*weight * cost_inner
        
        gradient_squared_main = gradient_squared_main+np.square(grad_main)

        gradient_squared_biases_main = grad_bias_main ** 2
        
        global_cost = 0.5 * cost

        
        return ((x[0][0]),(grad_main,grad_bias_main,gradient_squared_main,gradient_squared_biases_main,global_cost,1))

def udpate_parameters(x,result_dict,dict_key):
  
  if x[0] in dict_key:
    
    x[1]=(np.asarray(x[1]) - np.asarray(result_dict[x[0]][0]))
    x[1]=x[1].tolist()
    x[2]=float(x[2]-result_dict[x[0]][1])
    x[3]=(np.asarray(x[3]) - np.asarray(result_dict[x[0]][2]))
    x[3]=x[3].tolist()
    x[4]=float(x[4]-x[2]-result_dict[x[0]][3])
    
  return (x[0],x[1],x[2],x[3],x[4])

######################################################## MAIN CODE ################################################################

#Choose among the two corpus one is few line the w_spok_2012 is the big one.
corpus=sc.textFile("/FileStore/tables/25p5jhem1486569260328/corpus_test_glove_project.txt")
#corpus=sc.textFile("/FileStore/tables/diams6eg1488655835378/w_spok_2012.txt").map(lambda x : x.split(".")).flatMap(lambda x : x).filter(lambda x: x!='').map(removePunctuation)

#id2word=corpus.flatMap(remove_space_and_extract_word).map(lambda x : (x,1)).reduceByKey(lambda x,y:x+y).zipWithIndex().map(lambda x : (x[0][0],(x[1],x[0][1])))
corpus=corpus.map(lambda x : x.split(",")).flatMap(lambda x : x)


############ TRAINING PARAMETERS ###########@
min_count=2
vector_size=10
x_max=100
alpha=0.75
learning_rate=0.05
iterations=8

######## VOCAB SIZE AND ID2WORD #############
#id2word with frequency 
id2word=corpus.flatMap(remove_space_and_extract_word).map(lambda x : (x,1)).reduceByKey(lambda x,y:x+y).zipWithIndex().map(lambda x : (x[0][0],(x[1],x[0][1])))
id2word_braodcast=sc.broadcast(id2word.collectAsMap())
print(id2word_braodcast.value)
vocab_size=(len(id2word_braodcast.value))

#Return only word_id with frequence upper thant min_count
id2word_mincount=id2word.filter(lambda x: x[1][1] >= min_count).map(lambda x: x[1][0]).collect()

######## COMPUTE COOCURENCES #############

#Each sentence to token_ids
token_ids=corpus.map(lambda sentence : sentences_to_token_ids(sentence,id2word_braodcast))


coocurence_matrix_rdd=token_ids.map(compute_coocurence).map(lambda x: tuple(x)).flatMap(lambda x: x).reduceByKey(lambda x,y: x + y).filter(lambda x: (x[0][0] in id2word_mincount and x[0][1] in id2word_mincount)).map(lambda x: (x[0][0],x[0][1],x[1]))



######## INITIALISATION MODEL #############

#Initialise Weight, biases and gradient squared ... 
weight_and_biases_df = sc.parallelize(map(lambda x: build_data(x,vector_size),range(vocab_size))).toDF(['id_word','vector_represenation','biases','gradient_squared','gradient_squared_b'])
weight_and_biases_df.show()
#Coocur to dataframe 
coocurences_df = coocurence_matrix_rdd.toDF(['id_main','id_context','coocurence'])

for i in range(iterations):

  #Join over the coocurence and id_main, id_context to replace the id_main and context by their corresponding vector, biaises and gradient square 
  temp = coocurences_df.join(weight_and_biases_df, coocurences_df.id_main == weight_and_biases_df.id_word)\
        .select(weight_and_biases_df.vector_represenation.alias('main_vector'),weight_and_biases_df.biases.alias('biases_main')\
        ,weight_and_biases_df.gradient_squared.alias('gradient_squared_main'),weight_and_biases_df.gradient_squared_b.alias('gradient_squared_bm')\
        ,coocurences_df.id_context,coocurences_df.coocurence,coocurences_df.id_main)


  result = temp.join(weight_and_biases_df, temp.id_context == weight_and_biases_df.id_word)\
          .select(temp.main_vector,temp.biases_main,temp.gradient_squared_main,temp.gradient_squared_bm,weight_and_biases_df.vector_represenation.alias('context_vector')\
          ,weight_and_biases_df.biases.alias('biases_context')\
          ,weight_and_biases_df.gradient_squared.alias('gradient_squared_context')\
          ,weight_and_biases_df.gradient_squared_b.alias('gradient_squared_bc'),temp.coocurence,temp.id_main,temp.id_context).rdd\
    
    


  #Compute errors for eacht tuple of words and use reduceby key to do the average error. 
  result=result.map(lambda r: ((r.id_main,r.id_context),(r.main_vector,r.biases_main,r.gradient_squared_main,r.gradient_squared_bm,r.context_vector,r.biases_context
                                                         ,r.gradient_squared_context,r.gradient_squared_bc,r.coocurence)))\
         .map(lambda x: compute_error(x,learning_rate,x_max,alpha))\
         .reduceByKey(lambda a,b: ((np.sum([a[0],b[0]],axis=0),a[1]+b[1],np.sum([a[2],b[3]], axis=0),a[3]+b[3],a[4]+b[4],a[5]+b[5])))\
         .map(lambda x: (x[0],np.true_divide(x[1][0],x[1][5]),x[1][1]/x[1][5],np.true_divide(x[1][2],x[1][5]),x[1][3]/x[1][5],x[1][4]))\
         .map(lambda x: ((x[0],(x[1].tolist(),x[2],x[3].tolist(),x[4],x[5]))))
  
          
  global_cost=result.map(lambda x: (x[1][4])).sum()
  print(global_cost)


  result_dict = result.collectAsMap()

  dict_key=result_dict.keys()



  #Udpate the parameters 
  update=weight_and_biases_df.rdd.map(lambda r: ([r.id_word,r.vector_represenation,r.biases,r.gradient_squared,r.gradient_squared_b]))\
          .map(lambda x: udpate_parameters(x,result_dict,dict_key))


  weight_and_biases_df = update.toDF(['id_word','vector_represenation','biases','gradient_squared','gradient_squared_b'])


print('Last Update') 
weight_and_biases_df.show()

######################### INITIALISATION MODEL ##################



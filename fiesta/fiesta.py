'''
Module that contains the main fiesta functions.

Functions:

TTTS - Top-Two Thompson Sampling
'''

def TTTS() -> Tuple[]:
    '''
    This function requires as input at least some sort of List of method functions
    or perhaps more selectively a List of AllenNLP config path, then secondly 
    a data soucre of which we should allow them to give it optionally as a 
    train test split or not then lastly wether or not to 
    randomly split the data.

    Whether the model requires a development split is up to the method 
    functions
    :returns: Dict
    '''
    #require # models and desired confidence
    #initialize data storage (use lists because will be of different lengths)
    evaluations= [[] for i in range(N)]
    est_means=np.zeros(N)
    est_variances=np.zeros(N)
    #count# evals for each model
    eval_counts=np.zeros(N)
    #start by evaluating each model 3 times
    for i in range(0,N):
        for j in range(0,5):
            evaluations[i].append(eval_model(i))
        est_means[i]=np.mean(evaluations[i])
        est_variances[i]=np.var(evaluations[i],ddof=0)
        eval_counts[i]=len(evaluations[i])
    #initialize belief about location of best arm 
    pi=belief_calc(est_means,est_variances,eval_counts,100000*N)
    #run TTTS until hit required confidence
    #count number of evals
    num=3*N
    #store running counts of each arm pulled
    props=[]
    pis=[]
    while max(pi)<1-delta:
        props.append([x/sum(eval_counts) for x in eval_counts])
        pis.append(pi)
        #sample m-1
        m_1=np.random.choice(range(0,N), 1, p=pi)[0]
        r=np.random.uniform(0,1)
        if r<=0.5:
            #eval model m_1
            evaluations[m_1].append(eval_model(m_1))
            #update summary stats
            est_means[m_1]=np.mean(evaluations[m_1])
            est_variances[m_1]=np.var(evaluations[m_1],ddof=0)
            eval_counts[m_1]+=1 
            print("evalaution "+str(num)+" model "+str(m_1))
        else:
            #sample other model
            m_2=np.random.choice(range(0,N), 1, p=pi)[0]
            #resample until unique from model 1
            while m_1==m_2:
                m_2=np.random.choice(range(0,N), 1, p=pi)[0]
            #eval m_2
            evaluations[m_2].append(eval_model(m_2))
            #update summary stats
            est_means[m_2]=np.mean(evaluations[m_2])
            est_variances[m_2]=np.var(evaluations[m_2],ddof=0)
            eval_counts[m_2]+=1  
            print("evalaution "+str(num)+" model "+str(m_2))
        num+=1
        #update belief
        pi=belief_calc(est_means,est_variances,eval_counts,100000*N)
        print(pi)
    print("selected model "+str(np.argmax(pi)))
    return pi, props,pis,num
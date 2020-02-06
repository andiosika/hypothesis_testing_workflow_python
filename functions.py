
def Cohen_d(group1, group2, correction = False):
    """Compute Cohen's d
    d = (group1.mean()-group2.mean())/pool_variance.
    pooled_variance= (n1 * var1 + n2 * var2) / (n1 + n2)

    Args:
        group1 (Series or NumPy array): group 1 for calculating d
        group2 (Series or NumPy array): group 2 for calculating d
        correction (bool): Apply equation correction if N<50. Default is False. 
            - Url with small ncorrection equation: 
                - https://www.statisticshowto.datasciencecentral.com/cohens-d/ 
    Returns:
        d (float): calculated d value
         
    INTERPRETATION OF COHEN's D: 
    > Small effect = 0.2
    > Medium Effect = 0.5
    > Large Effect = 0.8
    
    """    
    import scipy.stats as stats
    import scipy   
    import numpy as np
    N = len(group1)+len(group2)
    diff = group1.mean() - group2.mean()

    n1, n2 = len(group1), len(group2)
    var1 = group1.var()
    var2 = group2.var()

    # Calculate the pooled threshold as shown earlier
    pooled_var = (n1 * var1 + n2 * var2) / (n1 + n2)
    
    # Calculate Cohen's d statistic
    d = diff / np.sqrt(pooled_var)
    
    ## Apply correction if needed
    if (N < 50) & (correction==True):
        d=d * ((N-3)/(N-2.25))*np.sqrt((N-2)/N)
    
    return d

#Your code here
def find_outliers_Z(data,col=None):
    """Use scipy to calcualte absoliute Z-scores 
    and return boolean series where True indicates it is an outlier

    Args:
        data (DataFrame,Series,or ndarray): data to test for outliers.
        col (str): If passing a DataFrame, must specify column to use.

    Returns:
        [boolean Series]: A True/False for each row use to slice outliers.
        
    EXAMPLE USE: 
    >> idx_outs = find_outliers_df(df,col='AdjustedCompensation')
    >> good_data = data[~idx_outs].copy()
    """
    
    from scipy import stats
    import numpy as np
    import pandas as pd

    if isinstance(data, pd.DataFrame):
        if col is None:
            raise Exception('If passing a DataFrame, must provide col=')
        else:
            data = data[col]
    elif isinstance(data,np.ndarray):
        data= pd.Series(data)

    elif isinstance(data,pd.Series):
        pass
    else:
        raise Exception('data must be a DataFrame, Series, or np.ndarray')
    
    z = np.abs(stats.zscore(data))
    idx_outliers = np.where(z>3,True,False)
    return idx_outliers

    
    
def find_outliers_IQR(data):
    """Use Tukey's Method of outlier removal AKA InterQuartile-Range Rule
    and return boolean series where True indicates it is an outlier.
    - Calculates the range between the 75% and 25% quartiles
    - Outliers fall outside upper and lower limits, using a treshold of  1.5*IQR the 75% and 25% quartiles.

    IQR Range Calculation:    
        res = df.describe()
        IQR = res['75%'] -  res['25%']
        lower_limit = res['25%'] - 1.5*IQR
        upper_limit = res['75%'] + 1.5*IQR

    Args:
        data (Series,or ndarray): data to test for outliers.

    Returns:
        [boolean Series]: A True/False for each row use to slice outliers.
        
    EXAMPLE USE: 
    >> idx_outs = find_outliers_df(df['AdjustedCompensation'])
    >> good_data = df[~idx_outs].copy()
    
    """
    import pandas as pd
    df_b=data
    res= df_b.describe()

    IQR = res['75%'] -  res['25%']
    lower_limit = res['25%'] - 1.5*IQR
    upper_limit = res['75%'] + 1.5*IQR

    idx_outs = (df_b>upper_limit) | (df_b<lower_limit)

    return idx_outs

def test_equal_variance(grp1,grp2, alpha=.05):
    stat,p = stats.levene(grp1,grp2)
    if p<alpha:
        print(f"Levene's test p value of {np.round(p,3)} is < {alpha}, therefore groups do NOT have equal variance.")
    else:
        print(f"Normal test p value of {np.round(p,3)} is > {alpha},  therefore groups DOES have equal variance.")
    return p

def test_normality(grp_control,col='BL',alpha=0.05):
    import scipy.stats as stats
    stat,p =stats.normaltest(grp_control[col])
    if p<alpha:
        print(f"Normal test p value of {np.round(p,3)} is < {alpha}, therefore data is NOT normal.")
    else:
        print(f"Normal test p value of {np.round(p,3)} is > {alpha}, therefore data IS normal.")
    return p

def test_assumptions(**kwargs):
    import scipy.stats as stats
    res= [['Test','Group','Stat','p','p<.05']]
    
    all_data = []
    for k,v in kwargs.items():
        try:
            all_data.append(v)
            stat,p =stats.normaltest(v)
            res.append(['Normality',k,stat,p,p<.05])
        except:
            res.append(['Normality',k,'err','err','err'])
        
    stat,p = stats.levene(*all_data)
    res.append(['Equal Variance','All',stat,p,p<.05]) 
    
    res=pd.DataFrame(res[1:],columns=res[0]).round(3)
    
    return res

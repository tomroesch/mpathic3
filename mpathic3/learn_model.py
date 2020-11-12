import mpathic3.utils as utils

def main(
    df,
    lm='IM',
    modeltype='MAT',
    LS_means_std=None,
    db=None,
    iteration=30000,
    burnin=1000,
    thin=10,
    runnum=0,
    initialize='LS',
    start=0,
    end=None,
    foreground=1,
    background=0,
    alpha=0,
    pseudocounts=1,
    test=False,
    drop_library=False,
    verbose=False):

    # Select target region
    df.loc[:,'seq'] = df.loc[:, 'seq'].str.slice(start, end)

    # Collapse identical sequences
    df = utils.collapse_further(df)

    # Make sure all counts are ints (don't know why yet)
    col_headers = utils.get_column_headers(df)
    df[col_headers] = df[col_headers].astype(int)

    df.reset_index(inplace=True,drop=True)

    # If length not given, take most abundant length
    if not end:
        seqL = utils.most_frequent([len(seq) for seq in df['seq']])
    else:
        seqL = end-start
    
    # Drop sequences with wrong length
    df = df[df['seq'].apply(len) == (seqL)]
    df.reset_index(inplace=True,drop=True)

    # Run inference
    if lm == 'IM':
        seq_mat, wtrow = utils.dataset2mutarray(df.copy(), modeltype)
        #this is also an MCMC routine, do the same as above.
        """
        if initialize == 'rand':
            if modeltype == 'MAT':
                emat_0 = utils.RandEmat(len(df[seq_col_name][0]),len(seq_dict))
            elif modeltype == 'NBR':
                emat_0 = utils.RandEmat(len(df['seq'][0])-1,len(seq_dict))
        elif initialize == 'LS':
            emat_cols = ['val_' + inv_dict[i] for i in range(len(seq_dict))]
            emat_0_df = main(df.copy(),lm='LS',modeltype=modeltype,alpha=alpha,start=0,end=None,verbose=verbose)
            emat_0 = np.transpose(np.array(emat_0_df[emat_cols]))   
            #pymc doesn't take sparse mat        
        emat = MaximizeMI_memsaver(
                seq_mat,df.copy(),emat_0,wtrow,db=db,iteration=iteration,burnin=burnin,
                thin=thin,runnum=runnum,verbose=verbose)"""
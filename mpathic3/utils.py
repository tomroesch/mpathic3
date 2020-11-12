import numpy as np


def collapse_further(df):
    '''take clipped df and then collapse it further'''
    # Find sequences column
    output_df = df.groupby('seq').sum()
    output_df = output_df.reset_index()
    #now reorder columns so we have 'ct' first and 'seq' last
    ct_columns = [x for x in df.columns if 'ct' in x]
    output_df = output_df[ct_columns + ['seq']]
    #The evaluated column will now be incorrect, so we should delete it.
    try:
        output_df = output_df.drop('val',axis=1)
    except:
        pass   
    return output_df


def get_column_headers(df):
    """Return column headers for all count colums in dataframe."""
    col_headers = [name for name in df.columns if 'ct' in name]              
    return col_headers


def most_frequent(List): 
    dict = {} 
    count, itm = 0, '' 
    for item in reversed(List): 
        dict[item] = dict.get(item, 0) + 1
        if dict[item] >= count : 
            count, itm = dict[item], item 
    return(itm) 


def seqs2array_for_matmodel(seq_list):
    """
    Converts a list of sequences (all of which must be the same length) to a numpy array to be used for matrix model evalution
    """

    num_chars = 4

    # Initialize matrix
    num_seqs = len(seq_list)
    seq_length = len(seq_list[0])
    mat = np.zeros([num_seqs,num_chars*seq_length])
    c_to_i_dict = {"A":0, "C": 1, "G": 2, "T": 3}
    # Fill matrix row by row
    for n, seq in enumerate(seq_list):
        for i, c in enumerate(seq):
            k = c_to_i_dict[c]
            mat[n,num_chars*i+k] = 1
    return mat


def dataset2mutarray(dataset_df, modeltype, chunksize=1000, rowsforwtcalc=100):

    # Determine seqtype, etc.
    seqcol = 'seq'

    # Compute the wt sequence
    rowsforwtcalc = min(rowsforwtcalc, dataset_df.shape[0])
    dataset_head_df = dataset_df.head(rowsforwtcalc)
    mut_df = profile_mut(dataset_head_df)
    wtseq = ''.join(list(mut_df['wt']))
    wtrow = seqs2array_for_matmodel([wtseq]).ravel().astype(bool)
    numfeatures = len(wtrow)

    # Process dataframe in chunks
    startrow = 0
    endrow = startrow+chunksize-1
    numrows = dataset_df.shape[0]

    # Fill in mutarray (a lil matrix) chunk by chunk
    mutarray_lil = lil_matrix((numrows,numfeatures),dtype=int)
    matrix_filled = False
    while not matrix_filled:

        if startrow >= numrows:
            matrix_filled = True
            continue
        elif endrow >= numrows:
            endrow = numrows-1
            matrix_filled = True

        # Compute seqarray
        seqlist = list(dataset_df[seqcol][startrow:(endrow+1)])
        seqarray = seqs2array_for_matmodel(seqlist)

        # Remove wt entries
        tmp = seqarray.copy()
        tmp[:,wtrow] = 0

        # Store results from this chunk
        mutarray_lil[startrow:(endrow+1),:] = tmp

        # Increment rows
        startrow = endrow+1
        endrow = startrow + chunksize - 1

    # Convert to csr matrix
    mutarray_csr = mutarray_lil.tocsr()

    # Return vararray as well as binary representation of wt seq
    return mutarray_csr, wtrow


def profile_mut(counts_df, err=False):
    """Compute mutation rate at each position."""

    # Record positions in new dataframe
    mut_df = counts_df[['pos']].copy()
    ct_cols = ["ct", "ct_0", "ct_1"]

    # Compute mutation rate across counts
    max_ct = counts_df[ct_cols].max(axis=1)
    sum_ct = counts_df[ct_cols].sum(axis=1)
    mut = 1.0 - (max_ct/sum_ct)
    mut_df['mut'] = mut

    # Computation of error rate is optional
    if err:
        mut_err = np.sqrt(mut*(1.0-mut)/sum_ct)
        mut_df['mut_err'] = mut_err

    # Compute WT base at each position
    mut_df['wt'] = 'X'
    for col in ct_cols:
        indices = (counts_df[col] == max_ct).values
        mut_df.loc[indices, 'wt'] = col.split('_')[1]

    return mut_df


def profile_ct(dataset_df, bin=None, start=0, end=None):
    """
    Computes character counts at each position

    Arguments:
        dataset_df (pd.DataFrame): A dataframe containing a valid dataset.
        bin (int): A bin number specifying which counts to use
        start (int): An integer specifying the sequence start position
        end (int): An integer specifying the sequence end position

    Returns:
        counts_df (pd.DataFrame): A dataframe containing counts for each nucleotide/amino acid character at each position. 
    """

    colname = 'seq'
    num_chars = 4

    total_seq_length = len(dataset_df[colname].iloc[0])

    alphabet = ["A", "C", "G", "T"]

    if end is None:
        end=total_seq_length

    # Set positions
    poss = pd.Series(range(start,end),name='pos')
    num_poss = len(poss)

    # Retrieve counts
    if bin is None:
        ct_col = 'ct'
    else:
        ct_col = 'ct_%d'%bin
    counts = dataset_df[ct_col]

    # Compute counts profile
    counts_array = np.zeros([num_poss,num_chars])
    counts_cols = ['ct_'+a for a in alphabet]
    for i,pos in enumerate(range(start,end)):
        char_list = dataset_df[colname].str.slice(pos,pos+1)
        counts_array[i,:] = [np.sum(counts[char_list==a]) for a in alphabet]
    temp_df = pd.DataFrame(counts_array,columns=counts_cols)
    counts_df = pd.concat([poss,temp_df],axis=1)

    # Validate as counts dataframe
    counts_df = qc.validate_profile_ct(counts_df,fix=True)
    return counts_df
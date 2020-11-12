import mpathic3.utils as utils
import mpathic.information
import pymc3 as pm
import scipy as sp
import numpy as np


def eval_modelmatrix_on_mutarray(modelmatrix, mutarray, wtrow):

    # Compute constant contribution to model prediciton
    modelmatrix_vec = modelmatrix.ravel()
    const_val = np.dot(wtrow, modelmatrix_vec)

    # Prepare matrix for scanning mutarray
    tmp_matrix = modelmatrix.copy()
    indices = wtrow.reshape(modelmatrix.shape).astype(bool)
    wt_matrix_vals = tmp_matrix[indices]
    tmp_matrix -= wt_matrix_vals[:, np.newaxis]
    modelmatrix_for_mutarray = csr_matrix(np.matrix(tmp_matrix.ravel()).T)

    # Compute values
    mutarray_vals = mutarray * modelmatrix_for_mutarray
    vals = const_val + mutarray_vals.toarray().ravel()
    return vals

def alt4(df, coarse_graining_level = 0.01):
    '''
    '''
    n_groups=500
    n_seqs = len(df.index)
    binheaders = utils.get_column_headers(df)
    n_batches = len(binheaders)
    cts_grouped = sp.zeros([n_groups, n_batches])
    group_num = 0
    frac_empty = 1.0
    
    # Copy dataframe
    tmp_df = df.copy(binheaders + ['val'])

    # Speed up computation by coarse-graining model predictions
    if coarse_graining_level:
        assert type(coarse_graining_level) == float
        assert coarse_graining_level > 0
        vals = tmp_df['val'].values
        scale = np.std(vals)
        coarse_vals = np.floor((vals / scale) / coarse_graining_level)
        tmp_df['val'] = coarse_vals
        grouped = tmp_df.groupby('val')
        grouped_tmp_df = grouped.aggregate(np.sum)
        grouped_tmp_df.sort_index(inplace=True)
    else:
        grouped_tmp_df = tmp_df
        grouped_tmp_df.sort_values(by='val',inplace=True)

    # Get count columns
    ct_df = grouped_tmp_df[binheaders].astype(float)
    cts_per_group = ct_df.sum(axis=0).sum()/n_groups

    # Histogram counts in groups. This is a bit tricky
    group_vec = np.zeros(n_batches)
    for i,row in ct_df.iterrows():
        row_ct_tot = row.sum()
        row_ct_vec = row.values
        row_frac_vec = row_ct_vec / row_ct_tot 

        while row_ct_tot >= cts_per_group*frac_empty:
            group_vec = group_vec + row_frac_vec*(cts_per_group*frac_empty)
            row_ct_tot -= cts_per_group*frac_empty

            # Only do once per group_num
            cts_grouped[group_num,:] = group_vec.copy() 

            # Reset for new group_num
            group_num += 1
            frac_empty = 1.0
            group_vec[:] = 0.0
        group_vec += row_frac_vec*row_ct_tot
        
        frac_empty -= row_ct_tot/cts_per_group
    if group_num == n_groups-1:
        cts_grouped[group_num,:] = group_vec.copy()
    elif group_num == n_groups:
        pass
    else:
        raise TypeError(\
            'group_num={} does not match n_groups={}'.format(group_num,n_groups))

    # Smooth empirical distribution with gaussian KDE
    f_reg = scipy.ndimage.gaussian_filter1d(cts_grouped, 0.04*n_groups, axis=0)

    # Return mutual information
    return information.mutualinfo(f_reg)


def MaximizeMI_memsaver(
        seq_mat,
        df,
        emat_0,
        wtrow,
        db=None,
        burnin=1000,
        iteration=30000,
        thin=10,
        runnum=0,
        verbose=False
        ):

    n_seqs = seq_mat.shape[0]
    with pm.Model() as model:

    @pymc.stochastic(observed=True,dtype=pd.DataFrame)
    def pymcdf(value=df):
        return 0
    @pymc.stochastic(dtype=float)
    def emat(p=pymcdf,value=emat_0):         
        p['val'] = numerics.eval_modelmatrix_on_mutarray(np.transpose(value),seq_mat,wtrow)                     
        MI = EstimateMutualInfoforMImax.alt4(p.copy())  # New and improved
        return n_seqs*MI
    if db:
        dbname = db + '_' + str(runnum) + '.sql'
        M = pymc.MCMC([pymcdf,emat],db='sqlite',dbname=dbname)
    else:
        M = pymc.MCMC([pymcdf,emat])
    M.use_step_method(stepper.GaugePreservingStepper,emat)

    if not verbose:
        M.sample = shutthefuckup(M.sample)

    M.sample(iteration,thin=thin)
    emat_mean = np.mean(M.trace('emat')[burnin:],axis=0)
    return emat_mean


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
        seq_mat, wtrow = utils.dataset2mutarray(df.copy())
        emat_0 = utils.RandEmat(len(df['seq'][0]), 4)     
        emat = MaximizeMI_memsaver(
                seq_mat,
                df.copy(),
                emat_0,
                wtrow,
                db=db,
                iteration=iteration,
                burnin=burnin,
                thin=thin,
                runnum=runnum,
                verbose=verbose
                )
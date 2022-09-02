import shap
import time
import scipy.special
import numpy as np
import itertools
from sklearn.preprocessing import StandardScaler

def powerset(iterable):
    s = list(iterable)
    return itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(len(s) + 1))


def shapley_kernel(M, s):
    if s == 0 or s==M:
        return 10000
    #return (M - 1) / (scipy.special.binom(M, s) * s * (M - s))
    #return scipy.special.factorial(s-1)*scipy.special.factorial(M-s-1)/scipy.special.factorial(M-1)
    return 1/((M-1)*scipy.special.binom(M-2,s-1))

def shapley_weights(M, s):
    if s > M - 1 or s < 0:
        return 0
    else:
        return 1 / (scipy.special.binom(M - 1, s) * M)


def build_sample(M,x_explain,reference,n_perms):
    n_samples = (M-1)*n_perms+2
    samples_ones = np.zeros((n_samples, M + 1))
    samples_values = np.zeros((n_samples, M))
    reg_weights = np.zeros(n_samples)
    shap_weights = np.zeros(n_samples)
    shap_weights2 = np.zeros(n_samples)
    shap_weights_builder = np.zeros(M)
    shap_weights2_builder = np.zeros(M)
    reg_weights_builder = np.zeros(M)
    for i in range(n_samples):
        samples_values[i, :] = reference
    for l in range(M):
        reg_weights_builder[l] = shapley_kernel(M, l)
        shap_weights_builder[l] = shapley_weights(M, l - 1)
        shap_weights2_builder[l] = shapley_weights(M, l)
    for iteration in range(n_perms):
        permutation = np.array(range(M))
        np.random.shuffle(permutation)
        for subset_size,feature in enumerate(permutation[:-1]):
            #print(iteration*M+subset_size)
            subset_id = iteration*(M-1)+subset_size+1
            samples_ones[subset_id,permutation[:subset_size+1]]=1
            samples_values[subset_id,permutation[:subset_size+1]]=x_explain[permutation[:subset_size+1]]
            reg_weights[subset_id] = reg_weights_builder[subset_size+1]
            shap_weights[subset_id] = shap_weights_builder[subset_size+1]
            shap_weights2[subset_id] = shap_weights2_builder[subset_size+1]
    samples_ones[:,-1] = 1
    samples_ones[-1,:] = 1
    samples_values[-1,:] = x_explain
    reg_weights[-1] = 10000
    reg_weights[0] = 10000
    shap_weights[-1] = 1/M
    shap_weights[0] = 1/M
    shap_weights2[-1] = 1/M
    shap_weights2[0] = 1/M
    return samples_ones,samples_values,reg_weights,shap_weights,shap_weights2

def compute_objective(S,y,phi):
    objective = 0
    M = int(np.shape(S)[1])
    for s,val in zip(S,y):
        size = np.sum(s)
        p_s = shapley_kernel(M,size)
        objective += p_s*(val-np.dot(s,phi))**2
    return objective


def kernel_shap(f,x,reference,M,batch_size=20000):
    weights = np.arange(1, M-1)
    weights = 1 / (weights * (M - weights))
    weights = weights / np.sum(weights)
    S = np.zeros((batch_size, M+1), dtype=bool)
    S[:,-1]=1
    V = np.zeros((batch_size,M))
    reg_weights = np.zeros(batch_size)
    shap_weights_i = np.zeros(batch_size)
    shap_weights = np.zeros(batch_size)

    for i in range(batch_size):
        V[i, :] = reference

    subset_sizes = np.random.choice(M - 2, size=batch_size,
                                        p=weights) + 1
    subset_sizes *= 0
    subset_sizes += 5
    for i,subset_size in enumerate(subset_sizes):
        inds = np.random.choice(M, size=subset_size, replace=True)
        #inds = np.where(np.random.randint(0,2,size=M,dtype=bool))
        S[i,inds] = 1
        V[i,inds] = x[inds]
        reg_weights[i] = shapley_kernel(M, subset_size)
        shap_weights_i[i] = shapley_weights(M, subset_size - 1)
        shap_weights[i] = shapley_weights(M, subset_size)
    y = f(V)

    q = np.random.rand(M + 1)
    unique, idx = np.unique(np.dot(S, q), return_index=True)
    S_unique = S[idx, :]
    y_unique = y[idx]
    reg_weights_unique = reg_weights[idx]

    empty_row = np.zeros((1, M+1), dtype=bool)
    empty_row[-1,-1]  = 1
    full_row = np.ones((1, M+1), dtype=bool)
    S_bar = np.concatenate((empty_row, S, full_row))
    S_bar_unique = np.concatenate((empty_row, S_unique, full_row))
    y_emptyset = f(np.zeros((1,M))+reference)
    y_fullset = f(x.reshape(1,-1))
    reg_weights_bar = np.concatenate(([10000000], reg_weights, [10000000]))
    reg_weights_bar_unique = np.concatenate(([10000000], reg_weights_unique, [10000000]))
    y_bar = np.concatenate((y_emptyset, y,y_fullset))
    y_bar_unique = np.concatenate((y_emptyset, y_unique,y_fullset))


    R=2*np.sum(1/np.arange(1,M-1))
    mu1 = np.dot(S_unique.T,np.diag(reg_weights_unique))
    print(np.sum(mu1,-1)/R)

    R_Q = 0
    for s in S:
        R_Q += shapley_kernel(M,np.sum(s))

    k_approx = 3
    X = np.zeros((2 ** M, M))
    for i, s in enumerate(powerset(range(M))):
        s = list(s)
        X[i, s] = 1

    subset_sizes = np.sum(X,-1)
    idx_approx = (subset_sizes<=k_approx)+(subset_sizes>=M-k_approx)

    mu1_Q = np.zeros(M)
    mu2_Q = np.zeros((M,M))

    S2 = 1-S.copy()
    #for q in np.concatenate((S,S2)):
    for q in S:
        size = np.sum(q[:-1])
        kernel = shapley_kernel(M,size)
        for i in range(M):
            if q[i]:
                mu1_Q += kernel

        pairs = itertools.combinations(np.where(q[:-1])[0], 2)
        for (i, j) in pairs:
            if q[i] and q[j]:
                mu2_Q[i,j] += kernel
                mu2_Q[j,i] += kernel
    mu1_Q_val = np.max(mu1_Q)
    mu2_Q_val = np.max(mu2_Q)

    #print(np.sum(mu1,-1))
    tmp = np.linalg.inv(np.dot(np.dot(S_bar.T, np.diag(reg_weights_bar)), S_bar))
    kernel_shap_approx = np.dot(tmp, np.dot(np.dot(S_bar.T, np.diag(reg_weights_bar)), y_bar))
    print("KernelSHAP Result:", kernel_shap_approx, np.sum(kernel_shap_approx))
    print("DIFF:",np.sum(np.abs(kernel_shap_approx[:-1]-reg_shap[:-1])))

    tmp_unique = np.linalg.inv(np.dot(np.dot(S_bar_unique.T, np.diag(reg_weights_bar_unique)), S_bar_unique))
    kernel_shap_unique = np.dot(tmp_unique, np.dot(np.dot(S_bar_unique.T, np.diag(reg_weights_bar_unique)), y_bar_unique))
    print("Unique KernelSHAP Result:", kernel_shap_unique, np.sum(kernel_shap_unique))
    print("DIFF:",np.sum(np.abs(kernel_shap_unique[:-1]-reg_shap[:-1])))

    new_shap =  (y_fullset-y_emptyset)/M + (np.dot(np.dot(S[:,:-1].T,np.diag(reg_weights))-np.dot(np.ones((M,batch_size)), np.sum(S[:,:-1],-1)*np.diag(reg_weights)/M), y))
    print("New SHAP Result:", new_shap,np.sum(new_shap))
    print("DIFF:",np.sum(np.abs(new_shap-reg_shap[:-1])))

    new_shap_unique =  (y_fullset-y_emptyset)/M + np.dot(np.dot(S_unique[:,:-1].T,np.diag(reg_weights_unique)),y_unique)-np.dot(np.dot(np.ones((M,len(y_unique))), np.sum(S_unique[:,:-1],-1)*np.diag(reg_weights_unique)/M), y_unique)
    print("New Unique SHAP Result:", new_shap_unique,np.sum(new_shap_unique))
    print("DIFF:",np.sum(np.abs(new_shap_unique-reg_shap[:-1])))

    reg_shap = kernel_shap_full(f,x,reference,M)
    print("True SHAP:",reg_shap)

    print(compute_objective(S[:,:-1],y,reg_shap[:-1]+reg_shap[-1]/M))
    print(compute_objective(S[:,:-1],y,kernel_shap_approx[:-1]+kernel_shap_approx[-1]/M))
    print(compute_objective(S[:,:-1],y,new_shap))
    print(compute_objective(S_unique[:,:-1],y,reg_shap[:-1]+reg_shap[-1]/M))
    print(compute_objective(S_unique[:,:-1],y,kernel_shap_unique[:-1]+kernel_shap_unique[-1]/M))
    print(compute_objective(S_unique[:,:-1],y,new_shap_unique))



def kernel_shap_k(f,x,reference,M,k_approx=3,batch_size=128):
    y_emptyset = f(np.zeros((1, M)) + reference)
    y_fullset = f(x.reshape(1, -1))

    S_full = np.zeros((2 ** M, M + 1))
    S_full[:, -1] = 1
    reg_weights_full = np.zeros(2 ** M)
    V_full = np.zeros((2 ** M, M))
    for i in range(2 ** M):
        V_full[i, :] = reference

    for i, s in enumerate(powerset(range(M))):
        s = list(s)
        V_full[i, s] = x[s]
        S_full[i, s] = 1
        reg_weights_full[i] = shapley_kernel(M, len(s))
    y_full = f(V_full)-y_emptyset

    subset_sizes_full = np.sum(S_full[:,:-1],-1)
    #idx = (subset_sizes_full <= k_approx) + (subset_sizes_full >= M-k_approx)
    idx = (subset_sizes_full <= k_approx)*(subset_sizes_full>0) + (subset_sizes_full<M)*(subset_sizes_full >= M - k_approx)
    S_k = S_full[idx,:]
    reg_weights_k = reg_weights_full[idx]
    V_k = V_full[idx,:]
    y_k = y_full[idx]


    if batch_size > 0:
        weights = np.arange(k_approx+1, M-k_approx)
        weights = 1 / (weights * (M - weights))
        weights = weights / np.sum(weights)
        S_sample = np.zeros((batch_size, M+1), dtype=bool)
        S_sample[:,-1]=1
        V_sample = np.zeros((batch_size,M))
        reg_weights_sample = np.zeros(batch_size)
        shap_weights_i = np.zeros(batch_size)
        shap_weights = np.zeros(batch_size)

        for i in range(batch_size):
            V_sample[i, :] = reference

        subset_sizes = np.random.choice(np.maximum(0,M - 1 - 2*k_approx), size=batch_size,
                                            p=weights) + 1 + k_approx
        #subset_sizes *= 0
        #subset_sizes += 5
        for i,subset_size in enumerate(subset_sizes):
            inds = np.random.choice(M, size=subset_size, replace=False)
            #inds = np.where(np.random.randint(0,2,size=M,dtype=bool))
            S_sample[i,inds] = 1
            V_sample[i,inds] = x[inds]
            reg_weights_sample[i] = shapley_kernel(M, subset_size)
            shap_weights_i[i] = shapley_weights(M, subset_size - 1)
            shap_weights[i] = shapley_weights(M, subset_size)
        if len(subset_sizes) > 0:
            y_sample = f(V_sample)-y_emptyset
        else:
            y_sample = []
        S = np.concatenate((S_k,S_sample))
        reg_weights = np.concatenate((reg_weights_k,reg_weights_sample))
        V = np.concatenate((V_k,V_sample))
        y = np.concatenate((y_k,y_sample))

    else:
        S = S_k.copy()
        reg_weights = reg_weights_k.copy()
        V = V_k.copy()
        y = y_k.copy()

    q = np.random.rand(M + 1)
    unique, idx = np.unique(np.dot(S, q), return_index=True)
    S_unique = S[idx, :]
    y_unique = y[idx]
    reg_weights_unique = reg_weights[idx]

    empty_row = np.zeros((1, M+1), dtype=bool)
    empty_row[-1,-1]  = 1
    full_row = np.ones((1, M+1), dtype=bool)
    S_bar = np.concatenate((empty_row,S, full_row))
    S_bar_unique = np.concatenate((empty_row,S_unique, full_row))

    reg_weights_bar = np.concatenate(([10000], reg_weights, [10000]))
    reg_weights_bar_unique = np.concatenate(([10000000], reg_weights_unique, [10000000]))
    y_bar = np.concatenate(([0],y,y_fullset-y_emptyset))
    y_bar_unique = np.concatenate(([0], y_unique,y_fullset-y_emptyset))

    mu1_Q = np.zeros(M)
    mu2_Q = np.zeros((M, M))
    # for q in np.concatenate((S,S2)):
    for q in S:
        size = np.sum(q[:-1])
        kernel = shapley_kernel(M, size)
        for i in range(M):
            if q[i]:
                mu1_Q[i] += kernel

        pairs = itertools.combinations(np.where(q[:-1])[0], 2)
        for (i, j) in pairs:
            if q[i] and q[j]:
                mu2_Q[i, j] += kernel
                mu2_Q[j, i] += kernel

    mu1_Q_val = np.mean(mu1_Q)
    mu2_Q_val = np.sum(mu2_Q)/(M*(M-1))
    mu_est_comparison = mu1_Q_val - mu2_Q_val
    H_d_1 = np.sum(1/np.arange(1,M-1))
    R=2*np.sum(1/np.arange(1,k_approx-1))

    S_sizes = np.sum(S[:,:-1],-1)
    S_unique_sizes = np.sum(S_unique[:,:-1],-1)

    mu_i = 0
    mu_i_j = 0
    mu_i_unique = 0
    mu_i_j_unique = 0
    for k in range(1,M):
        N_k = np.sum(S_sizes==k)
        N_unique_k = np.sum(S_unique_sizes==k)
        m_k = shapley_kernel(M,k)
        mu_i += m_k*N_k*k/M
        mu_i_j += m_k*N_k*k*(k-1)/(M*(M-1))
        mu_i_unique += m_k*N_unique_k*k/M
        mu_i_j_unique += m_k*N_unique_k*k*(k-1)/(M*(M-1))
    mu_est = mu_i - mu_i_j
    mu_unique_est = mu_i_unique - mu_i_j_unique


    reg_shap = kernel_shap_full(f,x,reference,M)


    tmp = np.linalg.inv(np.dot(np.dot(S_bar.T, np.diag(reg_weights_bar)), S_bar))
    kernel_shap_approx = np.dot(tmp, np.dot(np.dot(S_bar.T, np.diag(reg_weights_bar)), y_bar))
    print("KernelSHAP Result:", kernel_shap_approx, np.sum(kernel_shap_approx))
    print("DIFF:",np.sum(np.abs(kernel_shap_approx[:-1]-reg_shap[:-1])))

    tmp_unique = np.linalg.inv(np.dot(np.dot(S_bar_unique.T, np.diag(reg_weights_bar_unique)), S_bar_unique))
    kernel_shap_unique = np.dot(tmp_unique, np.dot(np.dot(S_bar_unique.T, np.diag(reg_weights_bar_unique)), y_bar_unique))
    print("Unique KernelSHAP Result:", kernel_shap_unique, np.sum(kernel_shap_unique))
    print("DIFF:",np.sum(np.abs(kernel_shap_unique[:-1]-reg_shap[:-1])))

    new_shap =  (y_fullset-y_emptyset)/M +1/(mu_est)*(np.dot(np.dot(S[:,:-1].T,np.diag(reg_weights))-np.dot(np.ones((M,np.shape(S)[0])), S_sizes*np.diag(reg_weights)/M), y))
    print("New SHAP Result:", new_shap,np.sum(new_shap))
    print("DIFF:",np.sum(np.abs(new_shap-reg_shap[:-1])))

    new_shap_unique =  (y_fullset-y_emptyset)/M + 1/mu_unique_est*(np.dot(np.dot(S_unique[:,:-1].T,np.diag(reg_weights_unique)),y_unique)-np.dot(np.dot(np.ones((M,len(y_unique))), S_unique_sizes*np.diag(reg_weights_unique)/M), y_unique))
    print("New Unique SHAP Result:", new_shap_unique,np.sum(new_shap_unique))
    print("DIFF:",np.sum(np.abs(new_shap_unique-reg_shap[:-1])))

    full_new_shap = (y_fullset - y_emptyset) / M + (np.dot(np.dot(S_full[1:-1, :-1].T, np.diag(reg_weights_full[1:-1])) - np.dot(np.ones((M, np.shape(S_full[1:-1,:])[0])),
                                                           np.sum(S_full[1:-1, :-1], -1) * np.diag(reg_weights_full[1:-1]) / M), y_full[1:-1]))
    print("New FULL SHAP Result:", full_new_shap, np.sum(full_new_shap))
    print("DIFF:", np.sum(np.abs(full_new_shap - reg_shap[:-1])))

    print("True SHAP:",reg_shap)

    print(compute_objective(S[:,:-1],y,reg_shap[:-1]))
    print(compute_objective(S[:,:-1],y,kernel_shap_approx[:-1]))
    print(compute_objective(S[:,:-1],y,new_shap))
    print(compute_objective(S_unique[:,:-1],y,reg_shap[:-1]+reg_shap[-1]/M))
    print(compute_objective(S_unique[:,:-1],y,kernel_shap_unique[:-1]+kernel_shap_unique[-1]/M))
    print(compute_objective(S_unique[:,:-1],y,new_shap_unique))


def kernel_shap_full(f, x, reference, M):
    X = np.zeros((2 ** M, M + 1))
    X[:, -1] = 1
    reg_weights = np.zeros(2 ** M)
    V = np.zeros((2 ** M, M))
    ws = {}
    for i in range(2 ** M):
        V[i, :] = reference

    for i, s in enumerate(powerset(range(M))):
        s = list(s)
        V[i, s] = x[s]
        X[i, s] = 1
        ws[len(s)] = ws.get(len(s), 0) + shapley_kernel(M, len(s))
        reg_weights[i] = shapley_kernel(M, len(s))
    y = f(V)
    tmp = np.linalg.inv(np.dot(np.dot(X.T, np.diag(reg_weights)), X))
    reg_shap = np.dot(tmp, np.dot(np.dot(X.T, np.diag(reg_weights)), y))

    return reg_shap


def kernel_shap_full_test(f, x, reference, M):
    X = np.zeros((2 ** M, M + 1))
    X[:, -1] = 1
    reg_weights = np.zeros(2 ** M)
    shap_weights = np.zeros(2 ** M)
    shap_weights2 = np.zeros(2 ** M)
    shap_weights3 = np.zeros(2 ** M)

    V = np.zeros((2 ** M, M))
    ws = {}
    for i in range(2 ** M):
        V[i, :] = reference

    for i, s in enumerate(powerset(range(M))):
        s = list(s)
        V[i, s] = x[s]
        X[i, s] = 1
        ws[len(s)] = ws.get(len(s), 0) + shapley_kernel(M, len(s))
        reg_weights[i] = shapley_kernel(M, len(s))
        shap_weights[i] = shapley_weights(M, len(s) - 1)
        shap_weights2[i] = shapley_weights(M, len(s))
    y = f(V)
    tmp = np.linalg.inv(np.dot(np.dot(X.T, np.diag(reg_weights)), X))
    reg_shap = np.dot(tmp, np.dot(np.dot(X.T, np.diag(reg_weights)), y))

    sizes = np.sum(X, -1)
    idx = (sizes<=3) + (sizes >= 7)
    new_shap =  (y[-1]-y[0])/M + np.dot(np.dot(X[idx,:-1].T,np.diag(reg_weights[idx]))-np.dot(np.ones((M,np.sum(idx))), np.sum(X[idx,:-1],-1)*np.diag(reg_weights[idx])/M), y[idx])

    tmp2 = np.linalg.inv(np.dot(np.dot(X[idx,:].T, np.diag(reg_weights[idx])), X[idx,:]))
    reg_shap2 = np.dot(tmp, np.dot(np.dot(X[idx,:].T, np.diag(reg_weights[idx])), y[idx]))



    for k in range(1, M):
        X_sum = np.sum(X[:, :-1], 1)
        index = (X_sum <= k)  + (X_sum==M)#+(X_sum>= M-k)
        n_obs = np.sum(index)
        r = np.array(range(1, 2 ** M - 1))
        np.random.shuffle(r)
        dummy_zeros = np.zeros(2 ** M, dtype=bool)
        dummy_zeros[r[:n_obs - 2]] = True
        index = dummy_zeros + (X_sum == M) + (X_sum == 0)

        X_idx,V_idx,reg_weights_idx,shap_weights_idx,shap_weights_2_idx = build_sample(M,x,reference,k*100)
        y_idx = f(V_idx)
        #X_idx = X[index, :]
        #V_idx = V[index, :]
        #y_idx = y[index]
        #shap_weights_idx = shap_weights[index]
        #shap_weights_2_idx = shap_weights2[index]
        #reg_weights_idxk = reg_weights[index]

        bias = (y[-1] - y[0]) / M

        weight_matrix_1 = np.dot(X_idx[1:-1, :-1].T, (reg_weights_idx[1:-1]))

        factor_list = []
        for (i,j) in itertools.combinations(range(M),2):
            X_ij = np.expand_dims(X_idx[1:-1,i]*X_idx[1:-1,j],1)
            weight_matrix_2 = np.dot(X_ij.T,reg_weights_idx[1:-1])
            q_i0 = weight_matrix_1[i]
            q_i0_j0 = weight_matrix_2[0]
            factor = q_i0 - q_i0_j0
            factor_list.append(factor)
            #print(i,j,factor)
        #factor = k/(M-1)
        factor = np.mean(np.array(factor_list))
        print(factor)

        shap_approx = np.dot(np.dot(X_idx[:, :-1].T, np.diag(shap_weights_idx + shap_weights_2_idx)) - shap_weights_2_idx, y_idx)
        shap_approx_raw = np.dot(np.dot(X_idx[1:-1, :-1].T, np.diag(shap_weights_idx[1:-1] + shap_weights_2_idx[1:-1])) - shap_weights_2_idx[1:-1], y_idx[1:-1])
        #shap_approx_corr = np.dot(np.dot(X_idx[1:-1, :-1].T,np.diag(shap_weights_idx[1:-1])),y_idx[1:-1]) - np.dot(np.dot(1-X_idx[1:-1, :-1].T,np.diag(shap_weights_2_idx[1:-1])),y_idx[1:-1])

        shap_approx_corrected = shap_approx_raw/factor + bias
        print("--------------------------------------------------------_")
        print("NOBS:", n_obs)
        print(k, "-approximation:", shap_approx,shap_approx_corrected)
        print("SUM", np.sum(shap_approx),np.sum(shap_approx_corrected))
        print("SUM DIFF", np.sum(np.abs(reg_shap[:-1] - shap_approx)),np.sum(np.abs(reg_shap[:-1] - shap_approx_corrected)))

        tmp_idxk = np.linalg.inv(np.dot(np.dot(X_idx.T, np.diag(reg_weights_idx)), X_idx))
        shap_approx_regk = np.dot(tmp_idxk, np.dot(np.dot(X_idx.T, np.diag(reg_weights_idx)), y_idx))
        print("KernelSHAP Result:", shap_approx_regk)
        print("SUM", np.sum(shap_approx_regk[:-1]))
        print("SUM DIFF", np.sum(np.abs(reg_shap - shap_approx_regk)))

    print("REG SHAP:", reg_shap)
    print("SUM", np.sum(reg_shap))
    print("Offset", f(np.zeros(M).reshape(1, -1)))
    print("Prediction", f(x.reshape(1, -1)))


X, y_unscaled = shap.datasets.diabetes()
# X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
scaler.fit(X)
scaler.transform(X)

#y = (y_unscaled-np.mean(y_unscaled))/np.std(y_unscaled)

from sklearn import linear_model, neural_network, ensemble

# model = linear_model.LinearRegression()
#model = neural_network.MLPRegressor(random_state=42)
model = ensemble.RandomForestRegressor(n_estimators=1000, max_depth=None, min_samples_split=2, random_state=42)
model.fit(X, y_unscaled)

x_explain = np.array(X.iloc[0, :])
M = np.shape(X)[1]
f = model.predict

#kernel_shap(f, x_explain, 0, M)


reference = np.zeros(M)
x = x_explain.copy()
import shap
explainer = shap.KernelExplainer(f, np.reshape(reference, (1, len(reference))))
shap_values = explainer.shap_values(x,nsamples=478,keep_index_ordered=True)
print("shap_values =", shap_values)
print("base value =", explainer.expected_value)

reg_shap = kernel_shap_full(f, x, reference, M)

print("DIFF:", np.sum(np.abs(shap_values - reg_shap[:-1])))


def evaluate_subsets(S_shap,M):
    mu1_Q = np.zeros(M)
    mu2_Q = np.zeros((M, M))
    for q in S_shap:
        size = np.sum(q)
        kernel = shapley_kernel(M, size)
        for i in range(M):
            if q[i]:
                mu1_Q[i] += kernel

        pairs = itertools.combinations(np.where(q)[0], 2)
        for (i, j) in pairs:
            if q[i] and q[j]:
                mu2_Q[i, j] += kernel
                mu2_Q[j, i] += kernel

    mu2_Q_mean = np.sum(mu2_Q)/(M*(M-1))
    np.fill_diagonal(mu2_Q,mu2_Q_mean)
    #print("STD:",np.std(mu1_Q),np.std(mu2_Q))
    return  mu1_Q,mu2_Q

def estimate_subset_bias(S_sizes,M):
    mu_est = 0
    for k in range(1,M):
        N_k = np.sum(S_sizes==k)
        m_k = shapley_kernel(M,k)
        mu_est += N_k*m_k*(k*(M-k))/(M*(M-1))
    return mu_est

def compute_shap_values(f,S_shap,y_shap,M,x,reference):
    S_sizes = np.sum(S_shap,-1)
    weights_shap = np.zeros(np.shape(S_shap)[0])
    for i,s in enumerate(S_shap):
        weights_shap[i] = shapley_kernel(M,np.sum(s))

    mu1_Q, mu2_Q = evaluate_subsets(S_shap,M)
    mu_est = estimate_subset_bias(S_sizes,M)

    y_fullset = f(x.reshape(1,-1))
    y_emptyset = f(reference.reshape(1,-1))
    #y_shap -= y_emptyset
    y_shap_unbiased = y_shap -  y_emptyset
    new_shap = (y_fullset-y_emptyset) / M + 1 / (mu_est) * (np.dot(
        np.dot(S_shap.T, np.diag(weights_shap)) - np.dot(np.ones((M, np.shape(S_shap)[0])),
                                                       S_sizes * np.diag(weights_shap) / M), y_shap_unbiased))

    mu1_Q_val = np.mean(mu1_Q)
    mu2_Q_val = np.sum(mu2_Q)/(M*(M-1))
    #print("Empirical weight biases:",mu1_Q_val-mu2_Q_val,mu_est)

    tmp = np.concatenate((np.zeros((1,M)),S_shap,np.ones((1,M))))
    S_shap_bar = np.zeros((np.shape(tmp)[0],M+1))
    S_shap_bar[:,-1]=1
    S_shap_bar[:,:-1] = tmp
    weights_shap_bar = np.concatenate(([10000],weights_shap,[10000]))
    y_shap_bar = np.concatenate((f(reference.reshape(1,-1)),y_shap,f(x.reshape(1,-1))))
    tmp = np.linalg.inv(np.dot(np.dot(S_shap_bar.T, np.diag(weights_shap_bar)), S_shap_bar))
    old_shap = np.dot(tmp, np.dot(np.dot(S_shap_bar.T, np.diag(weights_shap_bar)), y_shap_bar))

    return new_shap, old_shap




def sample_k_subsets(k,M,num_iter,budget):
    num_pairs = int(scipy.special.binom(M,2))
    S = np.zeros((num_iter * num_pairs, M))
    perm = np.array(range(M-2))
    counter = 0
    for q in range(num_iter):
        pairs = itertools.combinations(range(M), 2)
        for l,(i, j) in enumerate(pairs):
            var_range = list(range(M))
            var_range.remove(i)
            var_range.remove(j)
            np.random.shuffle(var_range)
            S[l+q*num_pairs,i] = 1
            S[l+q*num_pairs,j] = 1
            S[l+q*num_pairs,var_range[:k-2]] = 1
            counter += 1
        if counter>budget-num_pairs:
            break
    return S[:counter,:]

def generate_k_subsets(k,M,n_samples=10,balanced=True):
    counter_samples = 0
    counter_features = np.ones(M)
    S=np.zeros((n_samples,M))
    S_used = {}
    counter_runs=0
    while 1==1:
        counter_runs += 1
        if balanced:
            p = (1-(counter_features)/np.sum(counter_features))/np.sum(1-(counter_features)/np.sum(counter_features))
        else:
            p=np.ones(M)/M
        #print(counter_runs, p)
        subset = np.random.choice(M,size=k,replace=False,p=p)
        S_sample = np.zeros(M)
        S_sample[subset] = 1
        S_sample_tuple = tuple(S_sample)
        n_samples_cap = np.minimum(n_samples,scipy.special.binom(M,k))
        if S_sample_tuple not in S_used:
            S_used[S_sample_tuple]=1
            S[counter_samples,:] = S_sample
            counter_samples += 1
            counter_features[subset] += 1
            if counter_samples == n_samples_cap:
                break
    return S


def generate_predictions(f,S,M,x,reference):
    V = np.zeros(np.shape(S))
    for i in range(np.shape(S)[0]):
        V[i,:] = S[i,:]*x+(1-S[i,:])*reference
    return f(V)

def generate_subsets(M,budget,balanced=True):
    S_full = np.zeros((2 ** M, M))
    for i, s in enumerate(powerset(range(M))):
        s = list(s)
        S_full[i, s] = 1
    weights = np.arange(1,M)
    weights = 1 / (weights * (M - weights))
    weights = weights / np.sum(weights)
    max_comb = scipy.special.binom(M,np.arange(1,M))
    weighted_sample_sizes = np.array(np.minimum(weights*budget,max_comb),dtype=int)
    S_final = np.zeros((np.sum(weighted_sample_sizes),M))
    S_full_sizes = np.sum(S_full,-1)
    counter = 0
    for k in range(1,M):
        number_of_samples = np.int(weighted_sample_sizes[k-1])
        if  number_of_samples >= scipy.special.binom(M,k):
            S_k = S_full[S_full_sizes==k,:]
        else:
            S_k = generate_k_subsets(k,M,number_of_samples,balanced)
        S_final[counter:counter+np.shape(S_k)[0],:] = S_k
        #print(counter,np.shape(S_k)[0])
        counter += np.shape(S_k)[0]
    return S_final

def print_shap_difference(my_shap,old_shap,true_shap):
    print("New SHAP Result:", my_shap, np.sum(my_shap))
    print("DIFF:", np.sum(np.abs(my_shap - true_shap[:-1])))

    print("Old SHAP Result:", old_shap, np.sum(old_shap))
    print("DIFF:", np.sum(np.abs(old_shap[:-1] - true_shap[:-1])))


n_epochs=10
for i in range(n_epochs):
    n_trials = 600
    S_balanced = generate_subsets(M,n_trials)
    evaluate_subsets(S_balanced,M)
    y_balanced = generate_predictions(f,S_balanced,M,x_explain,reference)
    mu1_Q_bal,mu2_Q_bal = evaluate_subsets(S_balanced,M)
    my_shap_bal_tmp, old_shap_bal_tmp = compute_shap_values(f,S_balanced,y_balanced,M,x_explain,reference)
    if i == 0:
        my_shap_bal = my_shap_bal_tmp.copy()
        old_shap_bal = old_shap_bal_tmp.copy()
    else:
        my_shap_bal = np.vstack((my_shap_bal,my_shap_bal_tmp))
        old_shap_bal = np.vstack((old_shap_bal,old_shap_bal_tmp))
    #print_shap_difference(my_shap_bal_tmp,old_shap_bal_tmp,reg_shap)

    S_random = generate_subsets(M,n_trials,balanced=False)
    y_random = generate_predictions(f,S_random,M,x_explain,reference)
    mu1_Q,mu2_Q = evaluate_subsets(S_random,M)
    evaluate_subsets(S_random,M)
    my_shap_rand_tmp, old_shap_rand_tmp = compute_shap_values(f,S_random,y_random,M,x_explain,reference)
    if i == 0:
        my_shap_rand = my_shap_rand_tmp.copy()
        old_shap_rand = old_shap_rand_tmp.copy()
    else:
        my_shap_rand=np.vstack((my_shap_rand,my_shap_rand_tmp))
        old_shap_rand=np.vstack((old_shap_rand,old_shap_rand_tmp))
    #print_shap_difference(my_shap_rand_tmp,old_shap_rand_tmp,reg_shap)


mean_my_shap_bal = np.mean(my_shap_bal,0)
mean_old_shap_bal = np.mean(old_shap_bal,0)
std_my_shap_bal = np.std(my_shap_bal,0)
std_old_shap_bal = np.std(old_shap_bal,0)

mean_my_shap_rand = np.mean(my_shap_rand,0)
mean_old_shap_rand = np.mean(old_shap_rand,0)
print_shap_difference(mean_my_shap_bal, mean_old_shap_bal, reg_shap)
print_shap_difference(mean_my_shap_rand, mean_old_shap_rand, reg_shap)

#S_shap = explainer.maskMatrix
#y_shap = explainer.y
#q = np.random.rand(M)
#unique, idx = np.unique(np.dot(explainer.maskMatrix, q), return_index=True)
#S_unique = explainer.maskMatrix[idx, :]


#my_shap,old_shap = compute_shap_values(f,S_shap,y_shap,explainer.M,x,explainer.data.data)
#my_shap,old_shap = compute_shap_values(f,S[:,:-1],np.expand_dims(y,1),explainer.M,x,explainer.data.data)



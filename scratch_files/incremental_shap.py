import scipy
import numpy as np
import itertools

class incrementalSHAP:
    def __init__(self, M, y_fullset, y_emptyset):
        #n_features, f(x), f(reference)
        self.M = M
        self.y_fullset = y_fullset
        self.y_emptyset = y_emptyset
        #SHAP values
        self.mu = 0
        self.offset = (y_fullset-y_emptyset)/self.M
        self.partial_sum = np.zeros(M)
        #Lookup table
        self.weight_vector = np.zeros(M)
        for k in range(M):
            self.weight_vector[k] = self.shapley_kernel(k)


    def shapley_kernel(self,k):
        if k == 0 or k == self.M:
            return 10000
        return 1 / ((self.M - 1) * scipy.special.binom(self.M - 2, k - 1))

    def update_shap(self,feature_subset,prediction):
        k = int(np.sum(feature_subset,-1))
        weight = self.weight_vector[k]
        self.mu += weight*(k*(self.M-k))/(self.M*(self.M-1))
        self.partial_sum += weight*(prediction-self.y_emptyset)*(feature_subset-k/self.M)

    def powerset(self,iterable):
        s = list(iterable)
        return itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(len(s) + 1))

    def kernel_shap_full(self,f, x, reference):
        X = np.zeros((2 ** M, M + 1))
        X[:, -1] = 1
        reg_weights = np.zeros(2 ** M)
        V = np.zeros((2 ** M, M))
        ws = {}
        for i in range(2 ** M):
            V[i, :] = reference

        for i, s in enumerate(self.powerset(range(M))):
            s = list(s)
            V[i, s] = x[s]
            X[i, s] = 1
            ws[len(s)] = ws.get(len(s), 0) + self.shapley_kernel(len(s))
            reg_weights[i] = self.shapley_kernel(len(s))
        y = f(V)
        tmp = np.linalg.inv(np.dot(np.dot(X.T, np.diag(reg_weights)), X))
        reg_shap = np.dot(tmp, np.dot(np.dot(X.T, np.diag(reg_weights)), y))

        return reg_shap


    def generate_predictions(self,S,model,x_explain,reference):
        V = np.zeros(np.shape(S))
        for i in range(np.shape(S)[0]):
            V[i,:] = S[i,:]*x_explain+(1-S[i,:])*reference
        return model(V)

    def generate_k_subsets(self,k, n_samples=10, balanced=True):
        counter_samples = 0
        counter_features = np.ones(self.M)
        S = np.zeros((n_samples, self.M))
        S_used = {}
        counter_runs = 0
        while 1 == 1:
            counter_runs += 1
            if balanced:
                p = (1 - (counter_features) / np.sum(counter_features)) / np.sum(
                    1 - (counter_features) / np.sum(counter_features))
            else:
                p = np.ones(self.M) / self.M
            # print(counter_runs, p)
            subset = np.random.choice(self.M, size=k, replace=False, p=p)
            S_sample = np.zeros(self.M)
            S_sample[subset] = 1
            S_sample_tuple = tuple(S_sample)
            n_samples_cap = np.minimum(n_samples, scipy.special.binom(self.M, k))
            if S_sample_tuple not in S_used:
                S_used[S_sample_tuple] = 1
                S[counter_samples, :] = S_sample
                counter_samples += 1
                counter_features[subset] += 1
                if counter_samples == n_samples_cap:
                    break
        return S

    def generate_subsets(self,budget,balanced=True):
        S_full = np.zeros((2 ** self.M, self.M))
        for i, s in enumerate(self.powerset(range(self.M))):
            s = list(s)
            S_full[i, s] = 1
        weights = np.arange(1,M)
        weights = 1 / (weights * (self.M - weights))
        weights = weights / np.sum(weights)
        max_comb = scipy.special.binom(self.M,np.arange(1,self.M))
        weighted_sample_sizes = np.array(np.minimum(weights*budget,max_comb),dtype=int)
        S_final = np.zeros((np.sum(weighted_sample_sizes),self.M))
        S_full_sizes = np.sum(S_full,-1)
        counter = 0
        for k in range(1,self.M):
            number_of_samples = np.int(weighted_sample_sizes[k-1])
            if  number_of_samples >= scipy.special.binom(M,k):
                S_k = S_full[S_full_sizes==k,:]
            else:
                S_k = self.generate_k_subsets(k,number_of_samples,balanced)
            S_final[counter:counter+np.shape(S_k)[0],:] = S_k
            #print(counter,np.shape(S_k)[0])
            counter += np.shape(S_k)[0]
        return S_final

    def shapley_estimate(self):
        return self.offset + 1/self.mu*self.partial_sum


if __name__ == "__main__":
    from sklearn.preprocessing import StandardScaler
    from sklearn import linear_model, neural_network, ensemble
    import shap
    import numpy as np


    X, y_unscaled = shap.datasets.diabetes()

    scaler = StandardScaler()
    scaler.fit(X)
    scaler.transform(X)

    # y = (y_unscaled-np.mean(y_unscaled))/np.std(y_unscaled)
    # model = linear_model.LinearRegression()
    # model = neural_network.MLPRegressor(random_state=42)
    model = ensemble.RandomForestRegressor(n_estimators=1000, max_depth=None, min_samples_split=2, random_state=42)
    model.fit(X, y_unscaled)

    x_explain = np.array(X.iloc[0, :])
    M = np.shape(X)[1]
    f = model.predict
    # kernel_shap(f, x_explain, 0, M)
    reference = np.zeros(M)
    x = x_explain.copy()

    #explainer = shap.KernelExplainer(f, np.reshape(reference, (1, len(reference))))
    #shap_values = explainer.shap_values(x, nsamples=478, keep_index_ordered=True)
    #print("shap_values =", shap_values)
    #print("base value =", explainer.expected_value)

    prediction = f(x_explain.reshape(1,-1))
    offset = f(reference.reshape(1,-1))

    inc_shap = incrementalSHAP(M,prediction,offset)

    reg_shap = inc_shap.kernel_shap_full(f, x_explain, reference)

    S = inc_shap.generate_subsets(1500)
    S_pred = inc_shap.generate_predictions(S,f,x_explain,reference)

    for s,s_pred in zip(S,S_pred):
        inc_shap.update_shap(s,s_pred)
        shap_est = inc_shap.shapley_estimate()
        print(shap_est)

    print("DIFF:", np.sum(np.abs(shap_est - reg_shap[:-1])))

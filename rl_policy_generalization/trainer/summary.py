import numpy as np
from abc import ABC, abstractmethod
import torch
import torch.nn.functional as F

class Summarizer(ABC):
    # Borrowed from Borsos et al. (2020)

    def __init__(self, rs=None):
        super().__init__()
        if rs is None:
            rs = np.random.RandomState()
        self.rs = rs

    @abstractmethod
    def build_summary(self, X, y, size, **kwargs):
        pass

    def factory(type, rs):
        if type == 'uniform': return UniformSummarizer(rs)
        if type == 'kmeans_features': return KmeansFeatureSpace(rs)
        if type == 'kmeans_embedding': return KmeansEmbeddingSpace(rs)
        if type == 'kcenter_features': return KcenterFeatureSpace(rs)
        if type == 'kcenter_embedding': return KcenterEmbeddingSpace(rs)
        if type == 'icarl': return ICaRLSelection(rs)
        raise TypeError('Unkown summarizer type ' + type)

    factory = staticmethod(factory)

class UniformSummarizer(Summarizer):

    def build_summary(self, X, y, size, **kwargs):
        n = X.shape[0]
        inds = self.rs.choice(n, size, replace=False)
        return inds

class KmeansFeatureSpace(Summarizer):

    def kmeans_pp(self, X, k, rs):
        n = X.shape[0]
        inds = np.zeros(k).astype(int)
        inds[0] = rs.choice(n)
        dists = np.sum((X - X[inds[0]]) ** 2, axis=1)
        for i in range(1, k):
            ind = rs.choice(n, p=dists / np.sum(dists))
            inds[i] = ind
            dists = np.minimum(dists, np.sum((X - X[ind]) ** 2, axis=1))
        return inds

    def build_summary(self, X, y, size, **kwargs):
        X_flattened = X.reshape((X.shape[0], -1))
        inds = self.kmeans_pp(X_flattened, size, self.rs)
        return inds

class KmeansEmbeddingSpace(KmeansFeatureSpace):

    def get_embedding(self, X, model, device):
        embeddings = []
        with torch.no_grad():
            model.eval()
            for i in range(X.shape[0]):
                data = torch.from_numpy(X[i:i + 1]).float().to(device)
                embedding = model.embed(data)
                embeddings.append(embedding.cpu().numpy())
        return np.vstack(embeddings)

    def build_summary(self, X, y, size, **kwargs):
        embeddings = self.get_embedding(X, kwargs['model'], kwargs['device'])
        inds = self.kmeans_pp(embeddings, size, self.rs)
        return inds

# K-Center Coreset
class KcenterFeatureSpace(Summarizer):

    def update_distance(self, dists, x_train, current_id):
        for i in range(x_train.shape[0]):
            current_dist = np.linalg.norm(x_train[i, :] - x_train[current_id, :])
            dists[i] = np.minimum(current_dist, dists[i])
        return dists

    def kcenter(self, X, size):
        dists = np.full(X.shape[0], np.inf)
        current_id = 0
        dists = self.update_distance(dists, X, current_id)
        idx = [current_id]

        for i in range(1, size):
            current_id = np.argmax(dists)
            dists = self.update_distance(dists, X, current_id)
            idx.append(current_id)

        return np.hstack(idx)

    def build_summary(self, X, y, size, **kwargs):
        X_flattened = X.reshape((X.shape[0], -1))
        inds = self.kcenter(X_flattened, size)
        return inds

class KcenterEmbeddingSpace(KcenterFeatureSpace, KmeansEmbeddingSpace):

    def build_summary(self, X, y, size, **kwargs):
        model = kwargs['model']
        device = kwargs['device']
        embeddings = self.get_embedding(X, model, device)
        inds = self.kcenter(embeddings, size)
        return inds

# ICarl "moving-mean" sample selection in feature space, requires model
class ICaRLSelection(KmeansEmbeddingSpace):

    def build_summary(self, X, y, size, **kwargs):
        model = kwargs['model']
        device = kwargs['device']
        embeddings = self.get_embedding(X, model, device)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1)[:, np.newaxis]
        inds = []
        for c in np.unique(y):
            inds_c = []
            selected_inds = np.where(y == c)[0]
            target = np.mean(embeddings[selected_inds], axis=0)
            current_embedding = np.zeros(embeddings.shape[1])
            for i in range(size // len(np.unique(y)) + 1):
                best_score = np.inf
                for candidate in selected_inds:
                    if candidate not in inds_c:
                        score = np.linalg.norm(
                            target - (embeddings[candidate] + current_embedding) / (i + 1))
                        if score < best_score:
                            best_score = score
                            best_ind = candidate
                inds_c.append(best_ind)
                current_embedding = current_embedding + embeddings[best_ind]
            inds.append(inds_c)
        final_inds = []
        cnt = 0
        crt_pos = 0
        while cnt < size:
            for i in range(len(inds)):
                final_inds.append(inds[i][crt_pos])
                cnt += 1
                if cnt == size:
                    break
            crt_pos += 1
        inds = np.array(final_inds)
        return inds
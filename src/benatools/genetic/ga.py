import numpy as np
import pandas as pd


class GA():
    def __init__(self, seed=7, verbose=True, init_fn=None, init_kwargs={}):
        np.random.seed(seed)
        self.verbose = verbose

        self.init_fn = init_fn
        self.init_kwargs = init_kwargs

        self.generations = []
        self.scores = []
        
    def _generation(self, x, scores):
        x2 = self.selection(x.copy(), scores.copy())
        x2 = self.cross(x2)
        x2 = self.mutation(x2)
        x = self.replacement(x, scores, len(x2))
        return np.concatenate([x,x2])
    
    def run(self, n_gen, cost_function):
        print('Starting Genetic Algorithm for %i loops' % n_gen)
        
        history = []
        x = self.init_fn(**self.init_kwargs)
        scores = np.array([cost_function(e) for e in x])
        best_score = np.min(scores)
        best_sample = x[np.argmin(scores)]
        
        if self.verbose:
            print('\t Loop %i - Best %.6f - Avg %.6f - Worst %.6f' %(0, np.min(scores), np.mean(scores), np.max(scores) ))
        history.append({'best':np.min(scores), 'mean':np.mean(scores), 'worst':np.max(scores)})

        for i in range(0, n_gen):
            # Run transformations
            x = self._generation(x, scores)
            scores = np.array([cost_function(e) for e in x])

            # Save the best result
            if np.min(scores) < best_score:
                best_score = np.min(scores)
                best_sample = x[np.argmin(scores)]
            else:
            # Make sure the best is still alive. Replace the worst by the best
                x[np.argmax(scores)] = best_sample
                scores[np.argmax(scores)] = best_score

            # Save generation results
            self.generations.append(x.copy())
            self.scores.append(scores)

            # Print generation results
            if self.verbose:
                print('\t Loop %i - Best %.6f - Avg %.6f - Worst %.6f' %(i+1, np.min(scores), np.mean(scores), np.max(scores) ))
            history.append({'best':np.min(scores), 'mean':np.mean(scores), 'worst':np.max(scores)})
            
        self.best_sample = best_sample
        self.best_score = best_score
        
        return pd.DataFrame(history)
    
    def selection(self, x, scores):
        # Shuffle
        idx = np.arange(len(x))
        np.random.shuffle(idx)
        x = x[idx]
        scores = scores[idx]
        
        # Split and compare
        splits = scores.reshape(len(scores)//2,2)
        winners = splits.argmin(axis=1)
        x = x.reshape(2,len(x)//2,-1)
        x = np.stack([x[winners[i],i,:] for i in range(x.shape[1])])        

        return x

    def cross(self, x, prob=0.5):
        # Shuffle
        idx = np.arange(len(x))
        np.random.shuffle(idx)
        x = x[idx]

        # Split and compare
        pairs= x.reshape(2,len(x)//2,-1)
        mask = np.random.choice(a=[False, True], size=(pairs.shape[1], pairs.shape[2]))

        # Cross
        cross1 = np.where(mask, pairs[0,:], pairs[1,:] )
        cross2 = np.where(np.random.choice(a=[False, True], size=(pairs.shape[1], pairs.shape[2])), pairs[0,:], pairs[1,:] )
        cross = np.stack([cross1,cross2])

        # Decide
        return np.concatenate([cross[:,i,:] if x else pairs[:,i,:] for i,x in enumerate(np.random.choice(a=[False, True], size=(pairs.shape[1],), p=[1-prob, prob]))])
    
    def mutation(self, x, prob=0.05):
        inverse = np.logical_not(x)
        mask = np.random.choice(a=[False, True], size=x.shape, p=[1-prob, prob])
        x = np.where(mask, inverse, x)
        return x
    
    def replacement(self, x, scores, n_remove):
        # Shuffle
        idx = np.argsort(scores)
        x = x[idx]
        scores = scores[idx]
        x=x[:len(x)-n_remove]

        return x

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn import linear_model

class EvalModels:
    def __init__(self, train = None, test = None,
                 metric = mean_squared_error, n_splits = 5, 
                 cols = None, target = 'Target', id = 'Id'):

        if train is None:
            raise "The train dataset is nescessary"
        
        self.evaluations = {}
        self.train = train.copy()
        self.test = test.copy() if test else None

        self.cols = cols if cols else [col for col in train._get_numeric_data().columns if col != target]

        self.metric = metric
        self.target = target
        self.id = id
        self.kf = KFold(n_splits=n_splits, shuffle=True, random_state=2108)

    def eval(self, model = linear_model.LinearRegression()):
        evaluation = []
        train_predict = np.zeros(self.train.shape[0])
        if self.test:
            test_predict = np.zeros(self.test.shape[0])
        
        for train_index, test_index in self.kf.split(self.train):

            train_obs = self.train.loc[train_index,:]
            X_train = train_obs[self.cols]
            y_train = train_obs[self.target]

            model.fit(X_train, y_train)

            test_obs= self.train.loc[test_index,:]
            X_test = test_obs[self.cols]
            y_test = test_obs[self.target]

            evaluation.append(self.metric(y_test, model.predict(X_test)))

            train_predict += model.predict(self.train[self.cols])/self.kf.n_splits
        
            if self.test:
                test_predict += model.predict(self.test[self.cols])/self.kf.n_splits
        
        print('Score for {} model {:6.4f} ({:6.4f})'.format(model.__name__, np.mean(evaluation), np.std(evaluation)))

        ret = {'model': model,
               'metric': evaluation,
               'train_predict': train_predict,
               'test_predict': test_predict}

        self.evaluations[model.__name__] = ret
        
        return ret

    def eval_models(self, dict_of_models):
        '''
        {
            'lr': linear_model.LinearRegression(),
            'blr':linear_model.bayes.BayesianRidge(),
            'rfr':RandomForestRegressor(),
            'br': BaggingRegressor(),
            'abr':AdaBoostRegressor(),
            'gbr':GradientBoostingRegressor()
        }
        '''

        for name, model in dict_of_models.items():
            _ = self.eval(name, model)

    def to_csv(self, destransform = None):
        for model in self.evaluations.keys():
            pred = self.evaluations[model]['test_predict']
            destransform(pred) if destransform else pred

            pd.DataFrame({'Id': self.test[self.id], 
                          self.target: pred}).to_csv('./output/{:6.4f}_{:6.4f}_{}'.format(np.mean(self.evaluations[model]['metric']),
                                                                                          np.std(self.evaluations[model]['metric']),
                                                                                          model), index=False)

class Stack:
    def __init__(self, eval = None, use_feature = True):
        if eval is None:
            raise 'idiota'
            
        self.evaluations = {}
            
        self.train = eval.train
        self.test = eval.test
        self.metric = eval.metric
        self.target = eval.target
        self.kf = eval.kf

        dict_train = {}
        dict_test = {}

        for model in eval.evaluations.keys():
            aux = eval.evaluations[model]
            dict_train[model] = aux['train_predict']
            dict_test[model] = aux['test_predict']
        
        self.train_stack = pd.DataFrame(dict_train)
        self.test_stack  = pd.DataFrame(dict_test)

        cols_use = [col for col in eval.train._get_numeric_data() if col != self.target]

        if use_feature:
            self.train_stack = pd.concat([self.train_stack, eval.train[cols_use]], axis=1)
            self.test_stack  = pd.concat([self.test_stack,  eval.test._get_numeric_data()], axis=1)
        
    def eval(self,name_of_model = 'lr', model = linear_model.LinearRegression()):
        evaluation = []
        train_predict = np.zeros(self.train.shape[0])
        test_predict = np.zeros(self.test.shape[0])
        
        for train_index, test_index in self.kf.split(self.train):

            train_features_kf = self.train_stack.loc[train_index, :]
            train_target_kf   = self.train.loc[train_index, self.target]

            model.fit(train_features_kf, train_target_kf)

            test_features_kf = self.train_stack.loc[test_index, :]
            test_target_kf = self.train.loc[test_index,self.target]

            evaluation.append(self.metric(test_target_kf, model.predict(test_features_kf)))

            train_predict += model.predict(self.train_stack)/self.kf.n_splits
        
            if self.test is not None:
                test_predict += model.predict(self.test_stack)/self.kf.n_splits
        
        print('Score for {} model {:6.4f} ({:6.4f})'.format(name_of_model, np.mean(evaluation), np.std(evaluation)))

        ret = {'metric': evaluation,
               'train_predict': train_predict,
               'test_predict': test_predict}
        
        self.evaluations['stack_' + name_of_model] = ret
        
        return ret
    
        
    def eval_models(self, dict_of_models):
        '''
        {
                'lr': linear_model.LinearRegression(),
                'blr':linear_model.bayes.BayesianRidge(),
                'rfr':RandomForestRegressor(),
                'br': BaggingRegressor(),
                'abr':AdaBoostRegressor(),
                'gbr':GradientBoostingRegressor()
        }
        '''

        for name, model in dict_of_models.items():
            _ = self.eval(name, model)
            
    def to_csv(self, destransform = None):
        for model in self.evaluations.keys():
            if destransform:
                pred = destransform(self.evaluations[model]['test_predict'])
            else:
                pred = self.evaluations[model]['test_predict']

            pd.DataFrame({'Id': self.test.Id, 
                          self.target: pred}).to_csv('./output/{:6.4f}_{:6.4f}_{}'.format(np.mean(self.evaluations[model]['metric']),
                                                                                          np.std(self.evaluations[model]['metric']),
                                                                                          model), index=False)
import glob
import os
dataset_paths = sorted(glob.glob('datasets/*.csv'))
job_number = 1
classifiers = [
    'RandomForestClassifier',
    'AdaBoostClassifier',
    'BernoulliNB',
    'DecisionTreeClassifier',
    'ExtraTreesClassifier',
    'GaussianNB',
    'HistGradientBoostingClassifier',
    'KNeighborsClassifier',
    'LinearDiscriminantAnalysis',
    'LinearSVC',
    'SVC',
    'MLPClassifier',
    'MultinomialNB',
    'PassiveAggressiveClassifier',
    'QuadraticDiscriminantAnalysis',
    'SGDClassifier'
]

#classifiers = ['RandomForestClassifier']

for file_path in dataset_paths:
    for classificador in classifiers:
        job_name = f'job{job_number}'
        print(f'Sending {job_name} with parameters: Dataset = {file_path}')
        python_path = '' 
        time = '7-00:00:00'
        command = 
        #print(command)
        os.system(command)
        os.system('sleep 1') # pause to be kind to the scheduler
        job_number +=1
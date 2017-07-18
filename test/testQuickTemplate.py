from mlcore.engine.H3BaseClassifier import *
from mlcore.engine.TextClassifier import *

clf = TextClassifier()
x = np.random.rand(50,2)
y = np.random.choice([1,-1], 50)
clf.train(x,y)
clf.update(x,y)
clf.save('tmp.checkpoint')
xtt = np.random.rand(50,2)
ytt = np.random.choice([1,-1], 50)
clf2 = dill.load(open('tmp.checkpoint'))
y_pred = clf2.predict(xtt)
print 'mean error ', (ytt-y_pred).mean()
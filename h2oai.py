import h2o
h2o.init(start_h2o=True) 
train_url ="https://h2o-public-test-data.s3.amazonaws.com/bigdata/laptop/mnist/train.csv.gz"
test_url="https://h2o-public-test-data.s3.amazonaws.com/bigdata/laptop/mnist/test.csv.gz"


train=h2o.import_file(train_url)
test=h2o.import_file(test_url)

train.describe()
test.describe()


y='C785'
x=train.names[0:784]
train[y]=train[y].asfactor()
test[y]=test[y].asfactor()

from h2o.estimators.deeplearning import H2ODeepLearningEstimator

model_cv=H2ODeepLearningEstimator(distribution='multinomial'
                                 ,activation='RectifierWithDropout',hidden=[32,32,32],
                                        input_dropout_ratio=.2,
                                        sparse=True,
                                        l1=.0005,
                                            epochs=5)
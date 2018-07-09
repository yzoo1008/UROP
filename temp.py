from dataset import DataSet


train = DataSet('train')
test = DataSet('test')

print(train.get_size())
print(test.get_size())

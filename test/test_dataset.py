from drl_ids.datasets import KDDataset



train = KDDataset.load("./data", is_train=True)

print(train)

print(train[:10])

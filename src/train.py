from src.data.DARPA import DARPA_train, DARPA_test


method = "exact" # exact inference
N = 1
name = "DARPA_MSA_{}_{}".format(method, N)

train_set = DARPA_train
test_set = DARPA_test
### CONFIGS ###
dataset = 'citeseer'
model = 'VGAE'


input_dim = 3703  # 500#1433  # 500#3703#1433
hidden1_dim = 32
hidden2_dim = 16
hidden3_dim = 8
hidden_dims = [512, 256, 128, 64, 32, 16, 8]
num_feat_layers = len(hidden_dims)
output_dim = 4
use_feature = True

num_epoch = 3000
learning_rate = 3e-4

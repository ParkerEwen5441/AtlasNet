from numpy import load

data = load('/home/parker/datasets/TangConvNewTest/02691156/1a04e3eab45ca15dd86060f189eb133/scale_1.npz')
lst = data.files
# for item in lst:
#     print(item)
#     print(data[item])
    # input("WAIT")
print(data['points'].shape)
# print(data['nn_conv_ind'].shape)
print(data['pool_ind'].shape)

print(data['points'])
print(data['pool_ind'])

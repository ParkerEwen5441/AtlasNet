from numpy import load

data = load('/home/parker/datasets/ShapeNetTangConv/02828884/1NCK6OV5HW/tangent_image.npz')
lst = data.files
# for item in lst:
#     print(item)
#     print(data[item])
    # input("WAIT")
print(data['points'].shape)
print(data['pool_ind'].shape)
print(data['depth'].shape)


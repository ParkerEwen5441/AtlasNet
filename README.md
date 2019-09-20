🚀 Major upgrade 🚀 : Migration to  **Pytorch v1** and **Python 3.7**. The code is now much more generic and easy to install.

# AtlasNet Using Tangent Convolution Encoder

This repository contains a modified version of the AtlasNet network ([AtlasNet: A Papier-Mâché Approach to Learning 3D Surface Generation ](http://imagine.enpc.fr/~groueixt/atlasnet/)) which uses a portion of the Tangent Convolutions ([Tangent Convolutions for Dense Prediction in 3D](http://vladlen.info/papers/tangent-convolutions.pdf))network as an encoder.


### Citing this work

If you find this work useful in your research, please cite all papers:

```
@inproceedings{groueix2018,
          title={{AtlasNet: A Papier-M\^ach\'e Approach to Learning 3D Surface Generation}},
          author={Groueix, Thibault and Fisher, Matthew and Kim, Vladimir G. and Russell, Bryan and Aubry, Mathieu},
          booktitle={Proceedings IEEE Conf. on Computer Vision and Pattern Recognition (CVPR)},
          year={2018}
        }
@article{Tat2018,
  author    = {Maxim Tatarchenko* and Jaesik Park* and Vladlen Koltun and Qian-Yi Zhou.},
  title     = {Tangent Convolutions for Dense Prediction in {3D}},
  journal   = {CVPR},
  year      = {2018},
}
@article{Zhou2018,
  author    = {Qian-Yi Zhou and Jaesik Park and Vladlen Koltun},
  title     = {{Open3D}: {A} Modern Library for {3D} Data Processing},
  journal   = {arXiv:1801.09847},
  year      = {2018},
}
```

# Install

### Clone the repo and install dependencies

This implementation uses [Pytorch](http://pytorch.org/).

```shell
## Download the repository
git clone https://github.com/ThibaultGROUEIX/AtlasNet.git
cd AtlasNet
## Create python env with relevant packages
conda create --name pytorch-atlasnet python=3.7
source activate pytorch-atlasnet
pip install pandas visdom
# Install PyTorch as directed:
# https://pytorch.org/get-started/locally/?source=Google&medium=PaidSearch&utm_campaign=1712418477&utm_adgroup=66821158477&utm_keyword=%2Binstalling%20%2Bpytorch&utm_offering=AI&utm_Product=PYTorch&gclid=Cj0KCQjw5MLrBRClARIsAPG0WGwp-txrGdm03ajH11PmA_yzO-3KdxmFpal62fq5xajWiM6RETcL0l4aAg3NEALw_wcB
# you're done ! Congrats :)

```
# Training

### Data

```shell
cd data; ./download_data.sh; cd ..
```
We used the [ShapeNet](https://www.shapenet.org/) dataset for 3D models, and rendered views from [3D-R2N2](https://github.com/chrischoy/3D-R2N2):

When using the provided data make sure to respect the shapenet [license](https://shapenet.org/terms).

* [The point clouds from ShapeNet, with normals](https://cloud.enpc.fr/s/j2ECcKleA1IKNzk) go in ``` data/customShapeNet```
* [The corresponding normalized mesh (for the metro distance)](https://cloud.enpc.fr/s/RATKsfLQUSu0JWW) go in ``` data/ShapeNetCorev2Normalized```
* [the rendered views](https://cloud.enpc.fr/s/S6TCx1QJzviNHq0) go in ``` data/ShapeNetRendering```

The trained models and some corresponding results are also available online :

* [The trained_models](https://cloud.enpc.fr/s/c27Df7fRNXW2uG3) go in ``` trained_models/```

In case you need the results of ICP on PointSetGen output :
* [ICP on PSG](https://cloud.enpc.fr/s/3a7Xg9RzIsgmofw)



### Build chamfer distance (optional)

Using the custom chamfer distance will *divide memory usage by 2* and will be a bit faster. Use it if you're short on memory especially when training models for **Single View reconstruction**.

```shell
source activate pytorch-atlasnet
cd ./extension
python setup.py install
```



### Start training

* First launch a visdom server :

```bash
python -m visdom.server -p 8888
```

* Launch the training. Check out all the options in ```./training/train_AE_AtlasNet.py``` .

```shell
python ./training/train_TangConv_AtlasNet.py --env 'AE_AtlasNet' --nb_primitives 25
```

* Monitor your training on http://localhost:8888/

![visdom](pictures/visdom2.png)


* Compute some results with your trained model

  ```bash
  python ./inference/run_AE_AtlasNet.py
  ```
  The trained models accessible [here](TODO) have the following performances, slightly better than the one reported in [the paper](TODO). The number reported is the chamfer distance.


### Visualisation

The generated 3D models' surfaces are not oriented. As a consequence, some area will appear dark if you directly visualize the results in [Meshlab](http://www.meshlab.net/). You have to incorporate your own fragment shader in Meshlab, that flip the normals in they are hit by a ray from the wrong side. An exemple is given for the [Phong BRDF](https://en.wikipedia.org/wiki/Phong_reflection_model).

```shell
sudo mv /usr/share/meshlab/shaders/phong.frag /usr/share/meshlab/shaders/phong.frag.bak
sudo cp auxiliary/phong.frag /usr/share/meshlab/shaders/phong.frag #restart Meshlab
```

## Dependencies
Open3D from source
PyTorch

## Run Precompute Script
Change config/config.json file with input/output directories to fit user. Run utils/precompute.py

## Train Network

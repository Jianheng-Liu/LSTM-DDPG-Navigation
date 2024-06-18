# 代码文档
navigation项目工程文件代码文档

## 环境搭建
在Ubuntu18.04+ROS Melodic基础上配置强化学习环境  
本项目沿用turtlebot3 machine learning例程的python=2.7版本，可根据需求改为python=3.6+版本

+ 为同时使用ROS和Anaconda，配置相应依赖：
```shell
pip install -U rosinstall empy defusedxml netifaces
```
+ 安装cv_brige包，用于ROS中图像数据与opencv数据间的转换(python3需自行编译cv_brige包)
```shell
sudo apt-get install python-catkin-tools python-dev python-catkin-pkg-modules python-numpy python-yaml ros-melodic-cv-bridge
```
+ 安装opencv等其他所需包
```shell
pip install opencv-python numpy pyttsx visdom
```
+ DQN代码沿用turtlebot3 machine learning例程使用的python=2.7对应的CPU版本TensorFlow，若需GPU可自行配置其他版本
```shell
pip install --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.8.0-cp27-none-linux_x86_64.whl
pip install keras==2.1.5
pip install tensorboard
```
+ DDPG使用GPU版本的pytorch，暂未调试通
```shell
conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.1 -c pytorch
```

## 二维地图训练
### 地图创建
1. 将Cartographer建图得到的pgm格式地图存入文件夹`navigation/maps/cartpgrapher`
2. 运行`load_test.py`使用高斯滤波和泛洪填充得到训练地图，若改变地图创建效果可调整高斯滤波和泛洪填充参数，得到的训练地图存入文件夹`navigation/maps/environments`
3. 注意Cartographer保存的地图中一个像素点宽度对应真实空间中5cm
### DQN模型训练
1. `environment.py`为模型训练仿真环境，若`DISPLAY=True`则显示训练过程(用于调试验证)，若`DISPLAY_LASER=True`则显示激光雷达射线，其他参数可以调整机器人形状、机器人速度、仿真频率等  
2. `angents.py`中`class DQN()`为DQN算法模型
3. `dqn_train.py`为DQN模型训练程序，其中ENV_PATH配置使用的地图   
新建Terminal启动visdom.server可视化训练过程：
```shell
python -m visdom.server
```
创建新建文件夹`navigation/saved_models`保存训练模型权重，训练DQN模型:
```shell
python dqn_train.py
```
### DQN模型验证
将`environment.py`中`DISPLAY=True`显示环境  
在`dqn_inference`中`agent = DQN(state_size, action_size, load=, load_episode=)`设置用于推理的模型  
运行`dqn_inference`验证DQN模型：
```shell
python dqn_train.py
```

## 真实机器人推理
运行：
```shell
python nav.py
```

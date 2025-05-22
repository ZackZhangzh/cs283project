
# 🤖 LeapHand: L̲e̲arning A̲ P̲roficient Hand from Virtual to Reality

## Members

Zhihao Zhang 张智皓 \
2024291074 \
<zhangzhh12024@shanghaitech.com>

Kaichen Gong 龚开宸 \
2024233250 \
<gongkch2024@shanghaitech.com>

## 📄 Documentation

```bash
mkdir -p ~/cs283project/src
git clone git@github.com:ZackZhangzh/cs283project.git ~/cs283project/src
cd ~/cs283project

pip install -r requirements.txt
```

## 🦾 Run It

```bash
# ROS1 noetic
catkin_make
source devel/setup.bash
```

```bash

cd ~/cs283project/src/telekinesis
python arm_teleoperation.py 

```

```bash
roslaunch arm_teleoperation arm_teleoperation.launch
roslaunch arm_teleoperation arm_teleoperation.launch control_real_robot:=true


roslaunch kinova_bringup kinova_robot.launch 
```

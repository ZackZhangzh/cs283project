
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

```bash
# driver 
sudo cp src/kinova-ros/kinova_driver/udev/10-kinova-arm.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules && sudo udevadm trigger

lsusb | grep 22cd

sudo dmesg | grep -i kinova
```

## 🦾 Run It

```bash
# ROS1 noetic
catkin_make && source devel/setup.bash



```bash
COM13
roslaunch kinova_bringup kinova_robot.launch 

cd src/telekinesis
python src/telekinesis/leap_kinova_ik_real.py


python src/arm_teleoperation_real.py
```
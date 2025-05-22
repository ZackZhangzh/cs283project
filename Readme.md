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


conda create -n cs283 python==3.12
conda activate cs283
pip install -r requirements.txt
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
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
```

### Telekinesis Mode (Direct Control)

```bash
# Telekinesis mode - simple VR control of arm
# Uses toggle-based space key to control the arm
cd ~/cs283project/src/telekinesis
python arm_teleoperation_key.py 
```

### Full Teleoperation Mode

```bash
# Full teleoperation system with simulation
# This will prompt if you want to use keyboard control when Oculus is not available
roslaunch arm_teleoperation arm_teleoperation.launch

# Key Commands:
# - SPACE: Toggle control on/off
# - T: Toggle auto mode (move to red dot)
# - M: Toggle target movement mode (move red dot)
# - WASD/RF: Move in X/Y/Z directions
# - QEIJKL: Control orientation
# - SHIFT: Hold for faster movement
```

### Real Robot Control (Optional)

```bash
# Only use this if you have a real Kinova arm connected
roslaunch kinova_bringup kinova_robot.launch 
```

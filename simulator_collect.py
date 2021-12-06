from images import NumpyDatasetSaver, PandasSaver
from simulator import SimulatorMaster, FPS, ctrl2arr, get_controller, SimpleBullet, Point
import cv2
import numpy as np


nds = PandasSaver(save_dir='images/simulator_dataset', save_names=('player', 'bullet', 'ctrl'))
try:
    sm = SimulatorMaster()
    fps = FPS()
    counter = 0
    while True:
        if sm.hit:
            sm.__init__()
            nds.drop_last(120)
            counter = 0

        sm.bullets.append(SimpleBullet(
            point=Point(x=400, y=400),
            radius=abs(np.random.normal(scale=6)) + 5,
            theta=np.sin(counter / 25) * 180 + np.random.normal(scale=20, loc=90),
            speed=np.random.uniform(2.0, 4.0)))
        sm.bullets.append(SimpleBullet(
            point=Point(x=400, y=400),
            radius=abs(np.random.normal(scale=6)) + 5,
            theta=np.sin(counter / 25) * 180 + np.random.normal(scale=20, loc=90),
            speed=np.random.uniform(2.0, 4.0)))

        ctrl = get_controller()
        sm.control(ctrl=ctrl)
        counter += 1
        fps.step()
        sm.step()
        sm.draw_all()

        cv2.imshow('', sm.canvas)
        cv2.waitKey(1)

        nds.write_row(player=sm.draw_player_minimum(),
                      bullet=sm.draw_bullet_minimum(),
                      ctrl=ctrl2arr(ctrl=ctrl))

        while fps.get_fps() > 60:
            pass
        print(f'\rFPS: {fps} {len(nds.df.index)}', end=' ')
except (KeyboardInterrupt, FPS.TimerNotStartedError):
    nds.drop_last(120)
    nds.save_csv()

# Start this script
# sudo venv/bin/python3 simulator_collect.py

# Remove useless data
# sudo rm -r images/simulator_dataset/*

# Change data ownership
# sudo chown -R shin images/simulator_dataset/*

from simulator import SimulatorMaster, FPS, SimpleBullet, Point, arr2ctrl
import cv2
import numpy as np
from model import create_model
from images.image_utils import get_location, get_vision, draw_polar_lines

model = create_model()
model.load_weights('checkpoint/checkpoint-epoch015-loss0.153')

sm = SimulatorMaster()
fps = FPS()
counter = 0
while True:
    if sm.hit:
        sm.__init__()
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

    counter += 1
    fps.step()
    sm.step()
    sm.draw_all()

    data = {'bullet': sm.draw_bullet_minimum().astype(np.float32) / 255,
            'player': sm.draw_player_minimum().astype(np.float32) / 255}
    data['player_loc'] = np.array(get_location(data['player']))
    data['vision'] = get_vision(data['player_loc'], 256, data['bullet'])
    data['player_loc'][0] /= data['player'].shape[1]
    data['player_loc'][1] /= data['player'].shape[0]
    x_1, x_2 = np.expand_dims(data['vision'][..., 1:3], axis=0), np.expand_dims(data['player_loc'], axis=0)
    prediction = model((x_1, x_2), training=False)[0]
    prediction = np.round(prediction)
    ctrl = arr2ctrl(prediction)
    sm.control(ctrl=ctrl)

    sm.canvas = draw_polar_lines(sm.canvas,
                                 data['player_loc'],
                                 True,
                                 data['vision'],
                                 (255, 255, 255),
                                 1)
    sm.draw_all(reset_canvas=False)

    cv2.imshow('', sm.canvas)
    cv2.waitKey(1)

    while fps.get_fps() > 60:
        pass
    print(f'\rFPS: {fps}', end=' ')

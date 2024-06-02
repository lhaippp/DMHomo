import os
import cv2
import glob
import imageio

import numpy as np

def unit_test(im1_im2, homo12, name):
    im1_im2 = im1_im2.transpose(1, 2, 0)
    print(f"im1_im2 shape {im1_im2.shape}")
    img1 = im1_im2[..., :3]
    img2 = im1_im2[..., 3:]
    h, w, _ = img1.shape

    img1_warp = cv2.warpPerspective(img1, homo12, (w, h))

    buf1 = np.concatenate([img1, img1_warp], axis=1)[..., ::-1]
    buf2 = np.concatenate([img2, img2], axis=1)[..., ::-1]
    imageio.mimsave(f'unit_test/{name}.gif', [(buf1).astype(np.uint8), (buf2).astype(np.uint8)], duration=0.5, loop=0)


if __name__ == "__main__":
    npys = glob.glob('traindata/test/dataset/*npy*')
    idx = 0

    for npy in npys:
        print(f'process {npy}')

        # buf contains multiple lists, consisting of "imgs (N, 6, 256, 256)" and "homos (N, 3, 3)"
        buf = np.load(npy, allow_pickle=True)
        print(f'it contains {len(buf)} samples')
        is_head = True

        for _item in  buf:
            imgs = _item['imgs']
            homos = _item['homos']
            print(f'imgs shape {imgs.shape} | homos shape {homos.shape}')
            
            for i in range(len(imgs)):

                if is_head:
                    print('perform unit_test')
                    unit_test(imgs[i], homos[i], os.path.basename(npy))
                    is_head = False
                idx += 1
                print(idx)
                np.save(f'traindata/samples/{idx}.npy', {"img12": imgs[i], "homo12": homos[i]})

                # if idx >= 100:
                #     raise Exception('unit test')
            
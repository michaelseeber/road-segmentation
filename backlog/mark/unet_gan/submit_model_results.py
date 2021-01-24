import torch
import logging
from numpy import array, concatenate, newaxis, float64, save, load, uint8
import pickle
from os import listdir, mkdir
from os.path import join, exists
from PIL import Image
from torchvision import transforms

SESS_NAME = '20200511033356'
# MODEL_NAME = ['2020-5-10_2-18-12.028torch_model_CP_gen_epoch7.pth', '2020-5-10_2-18-12.028torch_model_CP_disc_epoch7.pth']
MODEL_NAME = ['20200511033356torch_model_CP_gen_epoch39.pth', '20200511033356torch_model_CP_disc_epoch39.pth']

DIR_TEST_IMAGES = '/Users/tamk/Documents/GitHub/cil-road-segmentation-2020/testing/images'
FP_TEST_IMAGES_AS_NPY = '/Users/tamk/Documents/GitHub/cil-road-segmentation-2020/testing/test_images_as_npy.npy'
FP_TEST_IMAGES_KEYS = '/Users/tamk/Documents/GitHub/cil-road-segmentation-2020/testing/test_images_keys.p'
DIR_SUBMISSION = '/Users/tamk/Documents/GitHub/cil-road-segmentation-2020/models/kmmt/submissions'
DIR_PREDICT = '/Users/tamk/Documents/GitHub/cil-road-segmentation-2020/models/kmmt/unet_gan/logs/' + SESS_NAME + '/PREDICTIONS'


print('='*80)
print('GATHERING TEST IMAGES')
print('='*80)

if False:

    files = sorted(listdir(DIR_TEST_IMAGES))
    ids = []

    for i, file in enumerate(files):
        ids.append(file.split('_')[1])

        im = Image.open(join(DIR_TEST_IMAGES, file))
        im = im.resize((256, 256))
        im = array(im, dtype=float).transpose(2, 0, 1)[newaxis, :, :, :] / 255
        if i == 0:
            x = im
        else:
            x = concatenate((x, im), axis=0)

    print(ids)
    pickle.dump(ids, open(FP_TEST_IMAGES_KEYS, "wb"))
    save(FP_TEST_IMAGES_AS_NPY, x)

print('='*80)
print('LOADING IMAGES')
print('='*80)

if True:
    x = load(FP_TEST_IMAGES_AS_NPY)
    ids = pickle.load(open(FP_TEST_IMAGES_KEYS, "rb"))


print('='*80)
print('PREDICTING')
print('='*80)

if True:
    x = torch.tensor(x, dtype=torch.float32)

    from core.models import ImageToBinary
    model_disc = ImageToBinary(n_channels_in=4)

    from core.models import UNet
    model_gen = UNet(n_channels=3, n_classes=1, bilinear=True)

    saved_models = ['/Users/tamk/Documents/GitHub/cil-road-segmentation-2020/models/kmmt/unet_gan/logs/' + SESS_NAME + '/CP/' + MODEL_NAME[0],
                    '/Users/tamk/Documents/GitHub/cil-road-segmentation-2020/models/kmmt/unet_gan/logs/' + SESS_NAME + '/CP/' + MODEL_NAME[1]]


    model_gen.load_state_dict(torch.load(saved_models[0], map_location=torch.device('cpu')))
    model_gen.eval()
    # model_disc.load_state_dict(torch.load(saved_models[1], map_location=torch.device('cpu')))

    batch_size = 4
    x = torch.split(x, batch_size)
    ids = [ids[i:i + batch_size] for i in range(0, len(ids), batch_size)]

    if not exists(DIR_PREDICT):
        mkdir(DIR_PREDICT)

    for _i, (_x, _ids) in enumerate(list(zip(x, ids))):
        _y_mask_fake = model_gen(_x)

        for i in range(_y_mask_fake.shape[0]):
            ar = _y_mask_fake[i, 0, :, :].detach().numpy() * 255
            im = Image.fromarray(ar.astype(uint8))
            im = im.convert("L")
            im = im.resize((608, 608))
            im.save(join(DIR_PREDICT, "test_{}".format(_ids[i])))


if True:
    from pathlib import Path
    import datetime
    import os

    filenames = [join(DIR_PREDICT, _f) for _f in os.listdir(DIR_TEST_IMAGES) if _f.startswith('test_')]
    print(filenames)
    from mask_to_submission import masks_to_submission

    # Path("submissions").mkdir(parents=True, exist_ok=True)
    submissions_filename = join(DIR_SUBMISSION, 'submission_' + SESS_NAME + '_' + datetime.datetime.now().strftime('%Y-%m-%d-%H-%M') + '.csv')
        
    masks_to_submission(submissions_filename, *filenames)


from os.path import join

DIR_TEST_IMAGES = '/Users/tamk/Google Drive/ETHZ/Course/[2020-T1]/[263-0008-00] Computational Intelligence Lab/Project/CIL Project Images/predictions_michi/merged/blend_pix2pix'
DIR_SUBMISSION = '/Volumes/DATABASE/c_s_ml/_train_data/misc/CIL/submissions'

DIR_TEST_IMAGES = '/Users/tamk/Downloads/Telegram Desktop/run2/predictions'
DIR_SUBMISSION = '/Users/tamk/Downloads/Telegram Desktop/run2/predictions'
if True:
    from pathlib import Path
    import datetime
    import os

    filenames = [join(DIR_TEST_IMAGES, _f) for _f in os.listdir(DIR_TEST_IMAGES) if not _f.startswith('.') and _f.endswith('.png')]
    print(filenames)

    from mask_to_submission import masks_to_submission

    # Path("submissions").mkdir(parents=True, exist_ok=True)
    submissions_filename = join(DIR_SUBMISSION, 'submission_' + datetime.datetime.now().strftime('%Y-%m-%d-%H-%M') + '.csv')
        
    masks_to_submission(submissions_filename, *filenames)
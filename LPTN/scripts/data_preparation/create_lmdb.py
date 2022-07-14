from codes.utils import scandir
from codes.utils.lmdb_util import make_lmdb_from_imgs


def create_lmdb():
    folder_path = '/home/jijang/projects/Drone/LPTN/datasets/disk/test_etc'     # image file path
    lmdb_path = '/home/jijang/projects/Drone/LPTN/datasets/lmdb/summer2winter/test_etc_etc.lmdb'    # lmdb file name
    img_path_list, keys = prepare_keys(folder_path)
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys)


def prepare_keys(folder_path):
    print('Reading image path list ...')
    img_path_list = sorted(
        list(scandir(folder_path, suffix='jpg', recursive=False)))
    keys = [img_path.split('.jpg')[0] for img_path in sorted(img_path_list)]

    return img_path_list, keys


if __name__ == '__main__':
    create_lmdb()

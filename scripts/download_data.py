import argparse
import json
import shutil
import zipfile
from io import BytesIO
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

FASHION_REVISION = '62b46580464039bdca8b68d323f631d40a4890af'
FASHION_VERSION = '3.0.0'


def download_text(root: Path) -> Path:
    path = root / 'data/raw/ruspellgold/test.jsonl'
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and path.stat().st_size > 0:
        print('RuSpellGold уже скачан.')
        return path

    downloaded = hf_hub_download(
        repo_id='RussianNLP/RuSpellGold',
        filename='data/complete_test/test.json',
        repo_type='dataset',
        revision='8b6694e45ac9222bc705ee4af9d6367ac083ff5c',
        local_dir=path.parent / '.hf',
        token=False,
    )
    temporary = path.with_suffix('.jsonl.tmp')
    try:
        shutil.copyfile(downloaded, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    print('RuSpellGold скачан.')
    return path


def download_fashion(root: Path) -> None:
    raw = root / 'data/raw/imaterialist'
    processed = root / 'data/processed'
    raw.mkdir(parents=True, exist_ok=True)
    processed.mkdir(parents=True, exist_ok=True)
    metadata_files = {
        'label_descriptions.json': raw / 'label_descriptions.json',
        'annotations.csv': processed / 'selected_annotations.csv',
        'image_manifest.csv': processed / 'image_manifest.csv',
        'classes.json': processed / 'classes.json',
        'dataset_info.json': processed / 'dataset_info.json',
    }
    images_ready = False
    if metadata_files['image_manifest.csv'].is_file():
        manifest = pd.read_csv(metadata_files['image_manifest.csv'], dtype={'ImageId': str})
        images_ready = len(manifest) == 4000 and manifest['ImageId'].is_unique
        images_ready = images_ready and all(
            (raw / 'train' / f'{image_id}.jpg').is_file()
            and (raw / 'train' / f'{image_id}.jpg').stat().st_size > 0
            for image_id in manifest['ImageId']
        )

    ready = images_ready and all(
        path.is_file() and path.stat().st_size > 0 for path in metadata_files.values()
    )
    if ready:
        info = json.loads(metadata_files['dataset_info.json'].read_text())
        ready = info.get('version') == FASHION_VERSION

    if ready:
        print('Фотографии и аннотации уже скачаны.')
        print('Готово фотографий:', len(manifest))
        return

    archive_path = hf_hub_download(
        repo_id='Monyzik/fashion-4000',
        filename='fashion-4000-metadata.zip' if images_ready else 'fashion-4000.zip',
        repo_type='dataset',
        revision=FASHION_REVISION,
        local_dir=raw / '.hf',
        token=False,
    )
    with zipfile.ZipFile(archive_path) as archive:
        manifest = pd.read_csv(
            BytesIO(archive.read('fashion-4000/image_manifest.csv')), dtype={'ImageId': str}
        )
        if len(manifest) != 4000 or not manifest['ImageId'].is_unique:
            raise ValueError('В архиве ожидались 4000 уникальных фото.')
        info = json.loads(archive.read('fashion-4000/dataset_info.json'))
        if info.get('version') != FASHION_VERSION:
            raise ValueError('В архиве найдена другая версия аннотаций.')

        def copy_member(member: str, target: Path) -> None:
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name + '.tmp')
            try:
                with archive.open(member) as source, temporary.open('wb') as destination:
                    shutil.copyfileobj(source, destination)
                if temporary.stat().st_size == 0:
                    raise ValueError(f'Пустой файл: {member}')
                temporary.replace(target)
            finally:
                temporary.unlink(missing_ok=True)

        metadata_files['dataset_info.json'].unlink(missing_ok=True)
        for member in ['label_descriptions.json', 'annotations.csv', 'classes.json']:
            copy_member(f'fashion-4000/{member}', metadata_files[member])
        for index, image_id in enumerate(manifest['ImageId'], 1):
            target = raw / 'train' / f'{image_id}.jpg'
            if not target.is_file() or target.stat().st_size == 0:
                copy_member(f'fashion-4000/images/{image_id}.jpg', target)
            if index % 1000 == 0:
                print(f'Подготовлено фото: {index}/4000')
        manifest['file_name'] = 'train/' + manifest['ImageId'] + '.jpg'
        temporary = processed / 'image_manifest.csv.tmp'
        manifest.to_csv(temporary, index=False)
        temporary.replace(metadata_files['image_manifest.csv'])
        copy_member('fashion-4000/dataset_info.json', metadata_files['dataset_info.json'])

    print('Готово фотографий:', len(manifest))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', choices=['text', 'fashion'])
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.dataset == 'text':
        download_text(root)
    else:
        download_fashion(root)


if __name__ == '__main__':
    main()

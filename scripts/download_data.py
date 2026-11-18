import argparse
import json
import shutil
import zipfile
from io import BytesIO
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

FASHION_REVISION = '4b3e23d4c0a7efe634b410bacbb44624871d314f'
FASHION_VERSION = '4.0.0'
FASHION_IMAGE_COUNT = 3922


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
    def image_path(image_id: str) -> Path:
        return raw / 'train' / f'{image_id}.jpg'

    def has_image(image_id: str) -> bool:
        path = image_path(image_id)
        return path.is_file() and path.stat().st_size > 0

    def valid_manifest(manifest: pd.DataFrame) -> bool:
        return len(manifest) == FASHION_IMAGE_COUNT and manifest['ImageId'].is_unique

    if all(path.is_file() and path.stat().st_size > 0 for path in metadata_files.values()):
        info = json.loads(metadata_files['dataset_info.json'].read_text())
        manifest = pd.read_csv(metadata_files['image_manifest.csv'], dtype={'ImageId': str})
        if (info.get('version') == FASHION_VERSION and valid_manifest(manifest)
                and all(map(has_image, manifest['ImageId']))):
            print(f'Набор {FASHION_VERSION} уже скачан: {len(manifest)} фото.')
            return

    def release_archive(filename: str) -> zipfile.ZipFile:
        return zipfile.ZipFile(hf_hub_download(
            repo_id='Monyzik/fashion-4000',
            filename=filename,
            repo_type='dataset',
            revision=FASHION_REVISION,
            local_dir=raw / '.hf',
            token=False,
        ))

    with release_archive('fashion-4000-metadata.zip') as archive:
        manifest = pd.read_csv(
            BytesIO(archive.read('fashion-4000/image_manifest.csv')), dtype={'ImageId': str}
        )
        if not valid_manifest(manifest):
            raise ValueError(f'В архиве ожидались {FASHION_IMAGE_COUNT} уникальных фото.')
        info = json.loads(archive.read('fashion-4000/dataset_info.json'))
        if info.get('version') != FASHION_VERSION:
            raise ValueError('В архиве найдена другая версия аннотаций.')

        def copy_member(source_zip: zipfile.ZipFile, member: str, target: Path) -> None:
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name + '.tmp')
            try:
                with source_zip.open(member) as source, temporary.open('wb') as destination:
                    shutil.copyfileobj(source, destination)
                if temporary.stat().st_size == 0:
                    raise ValueError(f'Пустой файл: {member}')
                temporary.replace(target)
            finally:
                temporary.unlink(missing_ok=True)

        missing = [image_id for image_id in manifest['ImageId'] if not has_image(image_id)]
        if missing:
            with release_archive('fashion-4000.zip') as images:
                for image_id in missing:
                    copy_member(images, f'fashion-4000/images/{image_id}.jpg', image_path(image_id))

        metadata_files['dataset_info.json'].unlink(missing_ok=True)
        for member in ['label_descriptions.json', 'annotations.csv', 'classes.json']:
            copy_member(archive, f'fashion-4000/{member}', metadata_files[member])
        manifest['file_name'] = 'train/' + manifest['ImageId'] + '.jpg'
        temporary = processed / 'image_manifest.csv.tmp'
        manifest.to_csv(temporary, index=False)
        temporary.replace(metadata_files['image_manifest.csv'])
        copy_member(archive, 'fashion-4000/dataset_info.json', metadata_files['dataset_info.json'])

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

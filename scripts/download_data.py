import argparse
import shutil
import zipfile
from io import BytesIO
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download


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
    }
    ready = all(path.is_file() and path.stat().st_size > 0 for path in metadata_files.values())
    if ready:
        manifest = pd.read_csv(metadata_files['image_manifest.csv'], dtype={'ImageId': str})
        ready = len(manifest) == 4000 and manifest['ImageId'].is_unique
        ready = ready and all(
            (raw / 'train' / f'{image_id}.jpg').is_file()
            and (raw / 'train' / f'{image_id}.jpg').stat().st_size > 0
            for image_id in manifest['ImageId']
        )

    if ready:
        print('Фотографии и аннотации уже скачаны.')
        print('Готово фотографий:', len(manifest))
        return

    archive_path = hf_hub_download(
        repo_id='Monyzik/fashion-4000',
        filename='fashion-4000.zip',
        repo_type='dataset',
        revision='0317dfeb42b06bcb0bbdf1d448bc20982653504d',
        local_dir=raw / '.hf',
        token=False,
    )
    with zipfile.ZipFile(archive_path) as archive:
        manifest = pd.read_csv(
            BytesIO(archive.read('fashion-4000/image_manifest.csv')), dtype={'ImageId': str}
        )
        if len(manifest) != 4000 or not manifest['ImageId'].is_unique:
            raise ValueError('В архиве ожидались 4000 уникальных фото.')

        def copy_member(member: str, target: Path) -> None:
            if target.is_file() and target.stat().st_size > 0:
                return
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

        for member in ['label_descriptions.json', 'annotations.csv']:
            copy_member(f'fashion-4000/{member}', metadata_files[member])
        for index, image_id in enumerate(manifest['ImageId'], 1):
            copy_member(f'fashion-4000/images/{image_id}.jpg', raw / 'train' / f'{image_id}.jpg')
            if index % 1000 == 0:
                print(f'Подготовлено фото: {index}/4000')
        manifest['file_name'] = 'train/' + manifest['ImageId'] + '.jpg'
        temporary = processed / 'image_manifest.csv.tmp'
        manifest.to_csv(temporary, index=False)
        temporary.replace(metadata_files['image_manifest.csv'])

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

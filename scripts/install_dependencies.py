from importlib import invalidate_caches, metadata
from pathlib import Path
import subprocess
import sys


def main() -> None:
    if sys.version_info < (3, 11):
        raise RuntimeError('Для проекта нужен Python 3.11 или новее.')

    import tomllib

    root = Path(__file__).resolve().parents[1]
    config = tomllib.loads((root / 'pyproject.toml').read_text(encoding='utf-8'))
    try:
        metadata.version('packaging')
    except metadata.PackageNotFoundError:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'packaging>=23'])
        invalidate_caches()
    from packaging.requirements import Requirement

    to_install = []
    to_update = []
    for specification in config['project']['dependencies']:
        requirement = Requirement(specification)
        if requirement.marker and not requirement.marker.evaluate():
            continue
        try:
            installed = metadata.version(requirement.name)
        except metadata.PackageNotFoundError:
            to_install.append(specification)
        else:
            if not requirement.specifier.contains(installed, prereleases=True):
                to_install.append(specification)
                to_update.append(requirement.name)

    if not to_install:
        print('Все зависимости уже установлены в подходящих версиях.')
        return

    print('Устанавливаем:', ', '.join(to_install), flush=True)
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', *to_install])
    print('Зависимости установлены.')
    if to_update:
        print('Обновлены:', ', '.join(to_update))
        print('Если эти библиотеки уже были импортированы, перезапустите ядро.')


if __name__ == '__main__':
    main()

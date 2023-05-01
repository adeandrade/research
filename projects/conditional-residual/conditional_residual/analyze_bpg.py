import os
import subprocess

import defopt
import torchvision.transforms.functional as functional
from PIL import Image
from sfu_torch_lib.file_fetcher import get_file_fetcher

import conditional_residual.functions as functions


def main(path_source: str, *, qp: int = 26) -> None:
    psnr, bpp, count = 0, 0, 0

    file_fetcher = get_file_fetcher(path_source, is_member=lambda member: member.endswith('.jpg'))

    path_bin = os.path.join(path_source, 'bin')
    path_destination = os.path.join(path_source, 'destination')

    os.makedirs(path_bin, exist_ok=True)
    os.makedirs(path_destination, exist_ok=True)

    for index in range(len(file_fetcher)):
        member = file_fetcher[index]

        prefix = member.removesuffix('.jpg')

        file_source = os.path.join(path_source, member)
        file_bin = os.path.join(path_bin, f'{prefix}.bin')
        file_destination = os.path.join(path_destination, f'{prefix}.png')

        subprocess.call(
            args=f'bpgenc -f 444 -q {qp} -o {file_bin} {file_source}'.split(' '),
            stdout=subprocess.DEVNULL,
        )
        subprocess.call(
            args=f'bpgdec -o {file_destination} {file_bin}'.split(' '),
            stdout=subprocess.DEVNULL,
        )

        with open(file_destination, 'rb') as object_destination:
            with file_fetcher.open_member(member) as object_source:
                image_destination = Image.open(object_destination).convert('RGB')
                image_source = Image.open(object_source).convert('RGB')

                tensor_destination = functional.to_tensor(image_destination)
                tensor_source = functional.to_tensor(image_source)

                _, psnr_instance = functions.calculate_reconstruction_loss(
                    predictions=tensor_destination[None],
                    targets=tensor_source[None],
                )

        psnr += psnr_instance
        bpp += os.stat(file_bin).st_size * 8 / tensor_source.shape[1] / tensor_source.shape[2]
        count += 1

    print(f'PSNR: {psnr / count}, BPP: {bpp / count}')


if __name__ == '__main__':
    defopt.run(main)

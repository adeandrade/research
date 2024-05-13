from compatible_representations.analyze_rate_distortion import Curve, Point, Plot


DEPTH_FOR_RECONSTRUCTION = Plot(
    distortion_label='PSNR',
    filename='depth_for_reconstruction',
    anchor='Baseline',
    experiments=(
        Curve(
            name='Proposed',
            metrics=['Validation Rate Distortion', 'Validation PSNR', 'Validation BPP'],
            run_ids=[
                'b68ef59e03cc4b98a497600db733d940',
                'f959f82dba5043158836aa3bbb8c7dd7',
                'b527cc2221ab48399d5ad4aedc73a93f',
                '399190d3f39748f9add1b09327a77290',
                '89ded9e7d82f48d28750bd715d1eebc7',
            ],
            mode='min',
            calculate_loss=lambda loss, _1, _2: loss,
            calculate_bpp=lambda _1, _2, bpp: bpp,
            calculate_distortion=lambda _1, psnr, _2: psnr,
        ),
        Curve(
            name='Baseline',
            metrics=['Validation Rate Distortion', 'Validation PSNR', 'Validation BPP'],
            run_ids=[
                'f762d2164cf746c0a586aba37f068f34',
                '50900475adbb470a9d19475b2c9947ba',
                'd3eaa643a11348d394ea37f8b24df33b',
                '3b2b6fab4610418b9d434a7a3075d3bf',
                'e6f051d4fcb041ce8ecae321e8a30f81',
            ],
            mode='min',
            calculate_loss=lambda loss, _1, _2: loss,
            calculate_bpp=lambda _1, _2, bpp: bpp,
            calculate_distortion=lambda _1, psnr, _2: psnr,
        ),
    ),
)


DEPTH_FOR_RECONSTRUCTION_DEPTH = Plot(
    distortion_label='RMSE',
    filename='depth_for_reconstruction_depth',
    anchor='Proposed',
    inverse=False,
    experiments=(
        Curve(
            name='Proposed',
            metrics=['Validation Rate Distortion', 'Validation Distortion Depth', 'Validation BPP'],
            run_ids=[
                'b68ef59e03cc4b98a497600db733d940',
                'f959f82dba5043158836aa3bbb8c7dd7',
                'b527cc2221ab48399d5ad4aedc73a93f',
                '399190d3f39748f9add1b09327a77290',
                '89ded9e7d82f48d28750bd715d1eebc7',
            ],
            mode='min',
            calculate_loss=lambda loss, _1, _2: loss,
            calculate_bpp=lambda _1, _2, bpp: bpp,
            calculate_distortion=lambda _1, distortion, _2: distortion,
        ),
        Curve(
            name='Baseline',
            metrics=['Validation Rate Distortion', 'Validation Distortion Depth', 'Validation BPP'],
            run_ids=[
                'f762d2164cf746c0a586aba37f068f34',
                '50900475adbb470a9d19475b2c9947ba',
                'd3eaa643a11348d394ea37f8b24df33b',
                '3b2b6fab4610418b9d434a7a3075d3bf',
                'e6f051d4fcb041ce8ecae321e8a30f81',
            ],
            mode='min',
            calculate_loss=lambda loss, _1, _2: loss,
            calculate_bpp=lambda _1, _2, bpp: bpp,
            calculate_distortion=lambda _1, distortion, _2: distortion,
        ),
    ),
    baseline=Point(
        run_id='f959f82dba5043158836aa3bbb8c7dd7',
        metrics=['Validation Distortion Depth', 'Validation BPP'],
        mode='min',
        calculate_loss=lambda distortion, _: distortion,
        calculate_bpp=lambda _, bpp: bpp,
        calculate_distortion=lambda distortion, _: distortion,
    ),
)


DEPTH_FOR_RECONSTRUCTION_SCALABLE = Plot(
    distortion_label='PSNR',
    filename='depth_for_reconstruction_scalable',
    anchor='Baseline',
    experiments=(
        Curve(
            name='Proposed',
            metrics=['Validation Loss', 'Validation PSNR', 'Validation BPP'],
            run_ids=[
                '77799004fc6644c19a78e22823bcd2d5',
                'bbb4efebca094bea8f159463339fddc4',
                'cad1c3b2eae643b4aff0e460308dde5b',
                'c641bb25d7d14c4e9321325274954507',
                'cdd9d1d9a101429bb9effa279320fb4d',
            ],
            mode='min',
            calculate_loss=lambda loss, _1, _2: loss,
            calculate_bpp=lambda _1, _2, bpp: bpp,
            calculate_distortion=lambda _1, psnr, _2: psnr,
        ),
        Curve(
            name='Baseline',
            metrics=['Validation Loss', 'Validation PSNR', 'Validation BPP'],
            run_ids=[
                '7e1ea1c6cb874734815d14113c715cc1',
                '854865f908504f7d9c52fe84e117ca97',
                'ec434333fbf041af98e187ea09970de9',
                '461e9c1d4b7c4b72aa51d01e6f6f4a82',
            ],
            mode='min',
            calculate_loss=lambda loss, _1, _2: loss,
            calculate_bpp=lambda _1, _2, bpp: bpp,
            calculate_distortion=lambda _1, psnr, _2: psnr,
        ),
        Curve(
            name='Standalone',
            metrics=['Validation PSNR', 'Validation BPP'],
            run_ids=[
                '2737c2c13d5c47efae49c36b2c414dce',
                '9e654d05591a47ca9da3f9d29337edd5',
                '2eca3d5063284b49a79850c874ae431b',
                '8bea8d02e30d42a7bdc09dd4898ec5d7',
                '0a9596ff0eb44b33af96bbc668f0e6dc',
            ],
            mode='min',
            calculate_loss=lambda _, bpp: bpp,
            calculate_bpp=lambda _, bpp: bpp,
            calculate_distortion=lambda psnr, _: psnr,
        ),
    ),
)


DETECTION_FOR_RECONSTRUCTION = Plot(
    distortion_label='PSNR',
    filename='detection_for_reconstruction',
    anchor='Baseline',
    experiments=(
        Curve(
            name='Proposed',
            metrics=['Validation Mean Average Precision', 'Validation PSNR', 'Validation BPP'],
            run_ids=[
                'ccef389cc7324f8bb094183b0b06dbe4',
                'a824b6e30b184ba0863eb9861c60b61c',
                'e36aa33dcba242b0a84cd8011857282b',
                'dd8253bf20274235ba7ee95fe0576b9c',
            ],
            mode='max',
            calculate_loss=lambda loss, _1, _2: loss,
            calculate_bpp=lambda _1, _2, bpp: bpp,
            calculate_distortion=lambda _1, psnr, _2: psnr + 12.9177,
        ),
        Curve(
            name='Baseline',
            metrics=['Validation Mean Average Precision', 'Validation PSNR', 'Validation BPP'],
            run_ids=[
                '2e505706a1554c7095c696dd8c81c09b',
                'e4e5ac2a03d549ceba5c62ef485c27e5',
                '3761fdfdceec48c398f6c4df2b115e83',
                '52b915c542fb407ebef45fc87c1b8240',
            ],
            mode='max',
            calculate_loss=lambda loss, _1, _2: loss,
            calculate_bpp=lambda _1, _2, bpp: bpp,
            calculate_distortion=lambda _1, psnr, _2: psnr + 12.9177,
        ),
    ),
)


DETECTION_FOR_RECONSTRUCTION_DETECTION = Plot(
    distortion_label='mAP',
    filename='detection_for_reconstruction_detection',
    anchor='Baseline',
    experiments=(
        Curve(
            name='Proposed',
            metrics=['Validation Mean Average Precision', 'Validation BPP'],
            run_ids=[
                'ccef389cc7324f8bb094183b0b06dbe4',
                'a824b6e30b184ba0863eb9861c60b61c',
                'e36aa33dcba242b0a84cd8011857282b',
                'dd8253bf20274235ba7ee95fe0576b9c',
            ],
            mode='max',
            calculate_loss=lambda loss, _: loss,
            calculate_bpp=lambda _, bpp: bpp,
            calculate_distortion=lambda distortion, _: distortion,
        ),
        Curve(
            name='Baseline',
            metrics=['Validation Mean Average Precision', 'Validation BPP'],
            run_ids=[
                '2e505706a1554c7095c696dd8c81c09b',
                'e4e5ac2a03d549ceba5c62ef485c27e5',
                '3761fdfdceec48c398f6c4df2b115e83',
                '52b915c542fb407ebef45fc87c1b8240',
            ],
            mode='max',
            calculate_loss=lambda loss, _: loss,
            calculate_bpp=lambda _, bpp: bpp,
            calculate_distortion=lambda distortion, _: distortion,
        ),
    ),
    baseline=Point(
        run_id='dd8253bf20274235ba7ee95fe0576b9c',
        metrics=['Validation Mean Average Precision', 'Validation BPP'],
        mode='max',
        calculate_loss=lambda loss, _: loss,
        calculate_bpp=lambda _, bpp: bpp,
        calculate_distortion=lambda distortion, _: distortion,
    ),
)


DETECTION_FOR_RECONSTRUCTION_SCALABLE = Plot(
    distortion_label='PSNR',
    filename='detection_for_reconstruction_scalable',
    anchor='Baseline',
    experiments=(
        Curve(
            name='Proposed',
            metrics=['Validation PSNR', 'Validation BPP'],
            run_ids=[
                '97658c0cd65f47869f174dbfc5807e2c',
                '97034dd5496144459329cbb856577950',
                '761a40b020c34b8ea7509fffbd34dac1',
                '2eb36580ff7d4000a67b53a467fe6251',
            ],
            mode='max',
            calculate_loss=lambda distortion, _: distortion,
            calculate_bpp=lambda _, bpp: bpp,
            calculate_distortion=lambda distortion, _: distortion + 12.9177,
        ),
        Curve(
            name='Baseline',
            metrics=['Validation PSNR', 'Validation BPP'],
            run_ids=[
                '26ef70a76e514ec385d89d530e6e2fd0',
                'f5603f82578b4b10b3f4f7902258761a',
                '8bebe2142bc64b4c9556e09a6f2022f8',
                '93e0ed9587c648ad8c96e55705684fb8',
            ],
            mode='max',
            calculate_loss=lambda distortion, _: distortion,
            calculate_bpp=lambda _, bpp: bpp,
            calculate_distortion=lambda distortion, _: distortion + 12.9177,
        ),
        Curve(
            name='Standalone',
            metrics=['Validation PSNR', 'Validation BPP'],
            run_ids=[
                'fce2ebc30a4045f389f64aadbe9e2cc2',
                '83eb8a7ae963495c8e064d43d8b16573',
                'dec288a2c9dc4bd5bae7fc67c0c3969f',
                'a8bd294ce116400a94ad9705fa9bf7c0',
            ],
            mode='min',
            calculate_loss=lambda _, bpp: bpp,
            calculate_bpp=lambda _, bpp: bpp,
            calculate_distortion=lambda distortion, _: distortion + 12.9177,
        ),
    ),
)


DEPTH_FOR_SEGMENTATION_SCALABLE = Plot(
    distortion_label='mIoU',
    filename='depth_for_segmentation_scalable',
    anchor='Baseline',
    experiments=(
        Curve(
            name='Proposed',
            metrics=['Train Segmentation IoU', 'Train BPP'],
            run_ids=[
                '61de1fb5239146d1b40d2f28468fdca7',
                '6b12be523f544770a2b74aa315e59289',
                '19381a23400e4c008831c177d085f2ad',
                '28d7b39079bf45ba951d1c01daedefb3',
                '72a27c1eee0246659181015c628be778',
            ],
            mode='max',
            calculate_loss=lambda iou, bpp: iou - bpp,
            calculate_bpp=lambda _, bpp: bpp,
            calculate_distortion=lambda iou, _: iou,
        ),
        Curve(
            name='Baseline',
            metrics=['Train Segmentation IoU', 'Train BPP'],
            run_ids=[
                '5b4df712facd4e038ecb87924818f532',
                'b82ed4c297314356a45c289d9c78601a',
                '9934047321994c849f3a7991dd537c67',
                '764bc6f0921a49538e72e84a2c7807c5',
                '4d7d53013f2f48d09c1881fd310cd4f0',
            ],
            mode='max',
            calculate_loss=lambda iou, _: iou,
            calculate_bpp=lambda _, bpp: bpp,
            calculate_distortion=lambda iou, _: iou,
        ),
        Curve(
            name='Standalone',
            metrics=['Train Segmentation IoU', 'Train BPP'],
            run_ids=[
                '876c80d88cd2410e91983d45c1eb33f5',
                '1600f012014047bf880f394d09253d91',
                '6656290d154e4d39806d32bccae2ef2e',
                'd51ac36ea6294f8791b35274833d6d56',
            ],
            mode='min',
            calculate_loss=lambda _, bpp: bpp,
            calculate_bpp=lambda _, bpp: bpp,
            calculate_distortion=lambda iou, _: iou,
        ),
    ),
)

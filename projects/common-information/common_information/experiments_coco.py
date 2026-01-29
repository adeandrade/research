from common_information.analyze_rate_distortion import CurveDistortions, Plot

MAIN = Plot(
    filename='coco-main',
    distortion_label='Detection + Keypointing mAP',
    anchor='Joint',
    normalize=False,
    baseline=0.455 + 0.65,
    experiments=[
        CurveDistortions(
            name='Joint',
            metrics=[
                'Validation Rate Distortion',
                'Validation BPP',
                'Validation Mean Average Precision Detection',
                'Validation Mean Average Precision Keypointing',
            ],
            run_ids=[
                'bdbf1834db644ee1868999d620c5d5fb',
                'bedf14cccca24d2ea3da370c2674a5d4',
                '8418a3798d8449cf894cee5cffee62b3',
                '5bf6cf92dec54be8bff155f86741ceb7',
            ],
            calculate_loss=lambda rd, r, d_a, d_b: rd,
            calculate_bpp=lambda rd, r, d_a, d_b: r,
            calculate_distortions=[
                lambda rd, r, d_a, d_b: d_a,
                lambda rd, r, d_a, d_b: d_b,
            ],
        ),
        CurveDistortions(
            name='Independent',
            metrics=[
                'Validation Rate Distortion',
                'Validation BPP A',
                'Validation BPP B',
                'Validation Mean Average Precision Detection',
                'Validation Mean Average Precision Keypointing',
            ],
            run_ids=[
                '7fd1d5a61c5a493683473c44f08a5369',
                'cfee0edb083c450d89f8f2b92be0bfc5',
                '447f6731e52c4a33b88a2071da1ff15c',
                'c811a379fdbc448380fdf88175152531',
                '6d4b095dd5d54623910ad30f573b4bb6',
            ],
            calculate_loss=lambda rd, r_a, r_b, d_a, d_b: rd,
            calculate_bpp=lambda rd, r_a, r_b, d_a, d_b: r_a + r_b,
            calculate_distortions=[
                lambda rd, r_a, r_b, d_a, d_b: d_a,
                lambda rd, r_a, r_b, d_a, d_b: d_b,
            ],
        ),
        CurveDistortions(
            name='Proposed (Transmit)',
            metrics=[
                'Validation Rate Distortion',
                'Validation BPP A',
                'Validation BPP B',
                'Validation BPP Common',
                'Validation Mean Average Precision Detection',
                'Validation Mean Average Precision Keypointing',
            ],
            run_ids=[
                'f779e9083076476785cae6061590c1ca',
                '1f6e72eebbf3417281fa93e3adf6fae6',
                'd28acf3282d14370b0bf910d5976a008',
                'aa8c880347cf4dcd8d0805cee122d0f2',
            ],
            calculate_loss=lambda rd, r_a, r_b, r_c, d_a, d_b: rd,
            calculate_bpp=lambda rd, r_a, r_b, r_c, d_a, d_b: r_a + r_b + r_c,
            calculate_distortions=[
                lambda rd, r_a, r_b, r_c, d_a, d_b: d_a,
                lambda rd, r_a, r_b, r_c, d_a, d_b: d_b,
            ],
        ),
        CurveDistortions(
            name='Proposed (Receive)',
            metrics=[
                'Validation Rate Distortion',
                'Validation BPP A',
                'Validation BPP B',
                'Validation BPP Common',
                'Validation Mean Average Precision Detection',
                'Validation Mean Average Precision Keypointing',
            ],
            run_ids=[
                'f779e9083076476785cae6061590c1ca',
                '1f6e72eebbf3417281fa93e3adf6fae6',
                'd28acf3282d14370b0bf910d5976a008',
                'aa8c880347cf4dcd8d0805cee122d0f2',
            ],
            calculate_loss=lambda rd, r_a, r_b, r_c, d_a, d_b: rd,
            calculate_bpp=lambda rd, r_a, r_b, r_c, d_a, d_b: r_a + r_b + 2 * r_c,
            calculate_distortions=[
                lambda rd, r_a, r_b, r_c, d_a, d_b: d_a,
                lambda rd, r_a, r_b, r_c, d_a, d_b: d_b,
            ],
        ),
    ],
)

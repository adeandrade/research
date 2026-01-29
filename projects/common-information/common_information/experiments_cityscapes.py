from common_information.analyze_rate_distortion import CurveDistortions, Plot

MAIN = Plot(
    filename='cityscapes-main',
    distortion_label='mIoU + Inverse RMSE',
    anchor='Joint',
    normalize=False,
    baseline=0.6716 + 1 / 5.5052,
    experiments=[
        CurveDistortions(
            name='Joint',
            metrics=[
                'Validation Rate Distortion',
                'Validation BPP',
                'Validation mIoU',
                'Validation Distortion Depth',
            ],
            run_ids=[
                '0f9d7781343a41948601ca24a4f0393c',
                'f76b2df33114463c9206b7e055caf66a',
                'de886ea954ac42d0b31891eea9c5e99b',
                '2698142d206f41fcb4899ea5b385930a',
                'daf876f3552f48fbaabd688819b18a35',
            ],
            calculate_loss=lambda rd, r, d_a, d_b: rd,
            calculate_bpp=lambda rd, r, d_a, d_b: r,
            calculate_distortions=[
                lambda rd, r, d_a, d_b: d_a,
                lambda rd, r, d_a, d_b: 1 / d_b,
            ],
        ),
        CurveDistortions(
            name='Independent',
            metrics=[
                'Validation Rate Distortion',
                'Validation BPP A',
                'Validation BPP B',
                'Validation mIoU',
                'Validation Distortion Depth',
            ],
            run_ids=[
                '4b52af03156d47e1a0568af75fda0d64',
                '4060c26c24e947a682fcc324a1f5e9a4',
                'ad276c60f9044d97ac88231545a22c0f',
                '8ed6d71d80a04a8c9cfcc1a72aa2f80c',
                'f1fe977ac18a4da79d5b63cd8cd830f5',
            ],
            calculate_loss=lambda rd, r_a, r_b, d_a, d_b: rd,
            calculate_bpp=lambda rd, r_a, r_b, d_a, d_b: r_a + r_b,
            calculate_distortions=[
                lambda rd, r_a, r_b, d_a, d_b: d_a,
                lambda rd, r_a, r_b, d_a, d_b: 1 / d_b,
            ],
        ),
        CurveDistortions(
            name='Proposed (Transmit)',
            metrics=[
                'Validation Rate Distortion',
                'Validation BPP A',
                'Validation BPP B',
                'Validation BPP Common',
                'Validation mIoU',
                'Validation Distortion Depth',
            ],
            run_ids=[
                'b2a4e1b56b464436bfce430e44cb37b8',
                'b60117590492444292cd30379fad74b6',
                'd12d2bec791d4c80a5155b17519d0401',
                '59aa10c45e92443e9b92f8d56e35ab0a',
                'ad84efbba5a04517870655c76e908129',
                '819358c9b4df497d9f0a8632de3dbe72',
            ],
            calculate_loss=lambda rd, r_a, r_b, r_c, d_a, d_b: rd,
            calculate_bpp=lambda rd, r_a, r_b, r_c, d_a, d_b: r_a + r_b + r_c,
            calculate_distortions=[
                lambda rd, r_a, r_b, r_c, d_a, d_b: d_a,
                lambda rd, r_a, r_b, r_c, d_a, d_b: 1 / d_b,
            ],
        ),
        CurveDistortions(
            name='Proposed (Receive)',
            metrics=[
                'Validation Rate Distortion',
                'Validation BPP A',
                'Validation BPP B',
                'Validation BPP Common',
                'Validation mIoU',
                'Validation Distortion Depth',
            ],
            run_ids=[
                'b2a4e1b56b464436bfce430e44cb37b8',
                'b60117590492444292cd30379fad74b6',
                'd12d2bec791d4c80a5155b17519d0401',
                '59aa10c45e92443e9b92f8d56e35ab0a',
                'ad84efbba5a04517870655c76e908129',
                '819358c9b4df497d9f0a8632de3dbe72',
            ],
            calculate_loss=lambda rd, r_a, r_b, r_c, d_a, d_b: rd,
            calculate_bpp=lambda rd, r_a, r_b, r_c, d_a, d_b: r_a + r_b + 2 * r_c,
            calculate_distortions=[
                lambda rd, r_a, r_b, r_c, d_a, d_b: d_a,
                lambda rd, r_a, r_b, r_c, d_a, d_b: 1 / d_b,
            ],
        ),
    ],
)

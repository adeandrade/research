import math

from lm_codec.analyze_rate_distortion import Curve, Plot, Point, PointData

LAMBDA = 0.01


RATE_DISTORTION_HYPER_PRIOR = Plot(
    distortion_label='Distortion (Perplexity)',
    rate_label='Rate (BPT)',
    legend_label='Method: BD-Rate (%)',
    filename='rate-distortion-hp',
    anchor='Deep Factorized Density Model',
    experiments=(
        Curve(
            name='Deep Factorized Density Model',
            metrics=['Validation Loss', 'Validation Distortion', 'Validation BPT'],
            run_ids=[
                'ebcd3aaabf014cfeaf2ff8337a143751',
                '762916ea115344889fde473dd01760b0',
                'b5203b6596b445b2b0392fef1ba626fa',
                '59b3ed1417e14f45a778d7c941b455f2',
            ],
            mode='min',
            calculate_loss=lambda _, distortion, bpt: distortion + LAMBDA * bpt,
            calculate_bpp=lambda _1, _2, bpt: bpt,
            calculate_distortion=lambda _1, distortion, _2: math.exp(distortion),
        ),
        Curve(
            name='Fourier Basis Density Model',
            metrics=['Validation Loss', 'Validation Distortion', 'Validation BPT'],
            run_ids=[
                'bce352cb58b2465f98dd45a3ef88b4b0',
                '3ddf59e610014ab282019e1fefe18fd4',
                '1e63445b2dc2472fbd6a30cffe399816',
                'ebe4f65c7ea449ca9fc3dc2abcf6b2d4',
            ],
            mode='min',
            calculate_loss=lambda _, distortion, bpt: distortion + LAMBDA * bpt,
            calculate_bpp=lambda _1, _2, bpt: bpt,
            calculate_distortion=lambda _1, distortion, _2: math.exp(distortion),
        ),
        Curve(
            name='Direct-Access Entropy Model',
            metrics=['Validation Loss', 'Validation Distortion', 'Validation BPT'],
            run_ids=[
                '899189ccde874b289979abf6c8632eae',
                '5f6b0680a3de42c1b93f70bf59a8526d',
                '623e78cd1f44422297a60c334a595a0f',
                '5f54eebb653240a285619de0d92f678c',
            ],
            mode='min',
            calculate_loss=lambda _, distortion, bpt: distortion + LAMBDA * bpt,
            calculate_bpp=lambda _1, _2, bpt: bpt,
            calculate_distortion=lambda _1, distortion, _2: math.exp(distortion),
        ),
    ),
    baseline_distortion=Point(
        run_id='719347766b65466ea4673d939e385234',
        metrics=['Testing Distortion'],
        mode='min',
        calculate_loss=lambda distortion: distortion,
        calculate_bpp=lambda _: None,
        calculate_distortion=lambda distortion: math.exp(distortion),
    ),
    baseline_bpp=PointData(bpp=768 * 16),
)


RATE_DISTORTION_LAYERS = Plot(
    distortion_label='Distortion (Perplexity)',
    rate_label='Rate (BPT)',
    filename='rate-distortion-layers',
    experiments=(
        Curve(
            name='Split Point 3',
            metrics=['Testing Loss', 'Testing Distortion', 'Testing BPT'],
            run_ids=[
                'fe26d51559bf4c7c9258c0eba1e44fd7',
                '9381e04c3037464099ec290947f52b66',
                'f174a93e57e347c4834d1294ec3404d6',
                'd8f98aaa2f6443019b14429937f4c934',
            ],
            mode='min',
            calculate_loss=lambda _, distortion, bpt: distortion + LAMBDA * bpt,
            calculate_bpp=lambda _1, _2, bpt: bpt,
            calculate_distortion=lambda _1, distortion, _2: math.exp(distortion),
        ),
        Curve(
            name='Split Point 6',
            metrics=['Validation Loss', 'Validation Distortion', 'Validation BPT'],
            run_ids=[
                'ebcd3aaabf014cfeaf2ff8337a143751',
                '762916ea115344889fde473dd01760b0',
                'b5203b6596b445b2b0392fef1ba626fa',
                '59b3ed1417e14f45a778d7c941b455f2',
            ],
            mode='min',
            calculate_loss=lambda _, distortion, bpt: distortion + LAMBDA * bpt,
            calculate_bpp=lambda _1, _2, bpt: bpt,
            calculate_distortion=lambda _1, distortion, _2: math.exp(distortion),
        ),
        Curve(
            name='Split Point 9',
            metrics=['Validation Loss', 'Validation Distortion', 'Validation BPT'],
            run_ids=[
                '1d5d580ae6f34d8c8672e12353657742',
                'cf358483b58041c7be3c4345e8e3c9a0',
                '2addca7c4e714be0a1ffe42f3265eab7',
                '30e91fe131084c9e88e7da526bdcda3d',
            ],
            mode='min',
            calculate_loss=lambda _, distortion, bpt: distortion + LAMBDA * bpt,
            calculate_bpp=lambda _1, _2, bpt: bpt,
            calculate_distortion=lambda _1, distortion, _2: math.exp(distortion),
        ),
    ),
    baseline_distortion=Point(
        run_id='719347766b65466ea4673d939e385234',
        metrics=['Testing Distortion'],
        mode='min',
        calculate_loss=lambda distortion: distortion,
        calculate_bpp=lambda _: None,
        calculate_distortion=lambda distortion: math.exp(distortion),
    ),
    baseline_bpp=PointData(bpp=768 * 16),
)


COVARIANCE_LAYER = Plot(
    distortion_label='Scaled Log Covariance Determinant',
    rate_label='Split Point',
    filename='layer-covariance',
    experiments=(
        Curve(
            name='GPT-2',
            metrics=['Covariance', 'Index'],
            run_ids=[
                'c4882b856b07487eb53238c776ad0979',
                '778eec98aa394a3888db67a4596dff8d',
                'd38c5fb7c0f54dcf90eb6e13ed48ffbb',
            ],
            calculate_loss=None,
            calculate_distortion=lambda covariance, _: covariance,
            calculate_bpp=lambda _, index: index,
        ),
        Curve(
            name='Pythia',
            metrics=['Covariance', 'Index'],
            run_ids=[
                '649c095e6a264baf9d3d60ea21b5dcea',
                '4c737b864bce4c18b16e7e0c16afa431',
                'b8b0148b12fc45a78a412912ea044cd4',
            ],
            calculate_loss=None,
            calculate_distortion=lambda covariance, _: covariance,
            calculate_bpp=lambda _, index: index,
        ),
        Curve(
            name='ViT',
            metrics=['Covariance', 'Index'],
            run_ids=[
                'bc485a43b8394bd8a6fd187388f54701',
                '05ae4381a17f46a78250d546df5ab9e7',
                'dfa9ad066ded45a887551f302a473e18',
            ],
            calculate_loss=None,
            calculate_distortion=lambda covariance, _: covariance,
            calculate_bpp=lambda _, index: index,
        ),
        Curve(
            name='ResNet',
            metrics=['Covariance', 'Index'],
            run_ids=[
                'e00e610a0f3041e9a3e770f047a2a0b4',
                'fe34ae6066524e06bd774759723a2d35',
                '943b3826e79e41fd98ae9ef6dba9cd7c',
            ],
            calculate_loss=None,
            calculate_distortion=lambda covariance, _: covariance,
            calculate_bpp=lambda _, index: resnet_index_to_layer(index),
        ),
    ),
)


COVARIANCE_BPT = Plot(
    distortion_label='Scaled Log Covariance Determinant',
    rate_label='Normalized Bitrate',
    legend_label='Method: Pearson Corr.',
    filename='bpt-covariance',
    pearson=True,
    normalize_rate=True,
    experiments=(
        Curve(
            name='GPT-2',
            metrics=['Covariance', 'BPT'],
            run_ids=[
                'c4882b856b07487eb53238c776ad0979',
                '778eec98aa394a3888db67a4596dff8d',
                'd38c5fb7c0f54dcf90eb6e13ed48ffbb',
            ],
            calculate_loss=None,
            calculate_distortion=lambda covariance, _: covariance,
            calculate_bpp=lambda _, bpt: bpt,
        ),
        Curve(
            name='Pythia',
            metrics=['Covariance', 'BPT'],
            run_ids=[
                '649c095e6a264baf9d3d60ea21b5dcea',
                '4c737b864bce4c18b16e7e0c16afa431',
                'b8b0148b12fc45a78a412912ea044cd4',
            ],
            calculate_loss=None,
            calculate_distortion=lambda covariance, _: covariance,
            calculate_bpp=lambda _, bpt: bpt,
        ),
        Curve(
            name='ViT',
            metrics=['Covariance', 'BPT'],
            run_ids=[
                'bc485a43b8394bd8a6fd187388f54701',
                '05ae4381a17f46a78250d546df5ab9e7',
                'dfa9ad066ded45a887551f302a473e18',
            ],
            calculate_loss=None,
            calculate_distortion=lambda covariance, _: covariance,
            calculate_bpp=lambda _, bpt: bpt,
        ),
        Curve(
            name='ResNet',
            metrics=['Covariance', 'BPT'],
            run_ids=[
                'e00e610a0f3041e9a3e770f047a2a0b4',
                'fe34ae6066524e06bd774759723a2d35',
                '943b3826e79e41fd98ae9ef6dba9cd7c',
            ],
            calculate_loss=None,
            calculate_distortion=lambda covariance, _: covariance,
            calculate_bpp=lambda _, bpt: bpt,
        ),
    ),
)


BPT_LAYER = Plot(
    distortion_label='Normalized Bitrate',
    rate_label='Split Point',
    filename='layer-bpt',
    normalize_distortion=True,
    normalize_rate=False,
    experiments=(
        Curve(
            name='GPT-2',
            metrics=['Index', 'BPT'],
            run_ids=[
                'c4882b856b07487eb53238c776ad0979',
                '778eec98aa394a3888db67a4596dff8d',
                'd38c5fb7c0f54dcf90eb6e13ed48ffbb',
            ],
            calculate_loss=lambda *_: 0.0,
            calculate_distortion=lambda _, bpt: bpt,
            calculate_bpp=lambda index, _: index,
        ),
        Curve(
            name='Pythia',
            metrics=['Index', 'BPT'],
            run_ids=[
                '649c095e6a264baf9d3d60ea21b5dcea',
                '4c737b864bce4c18b16e7e0c16afa431',
                'b8b0148b12fc45a78a412912ea044cd4',
            ],
            calculate_loss=lambda *_: 0.0,
            calculate_distortion=lambda _, bpt: bpt,
            calculate_bpp=lambda index, _: index,
        ),
        Curve(
            name='ViT',
            metrics=['Index', 'BPT'],
            run_ids=[
                'bc485a43b8394bd8a6fd187388f54701',
                '05ae4381a17f46a78250d546df5ab9e7',
                'dfa9ad066ded45a887551f302a473e18',
            ],
            calculate_loss=lambda *_: 0.0,
            calculate_distortion=lambda _, bpt: bpt,
            calculate_bpp=lambda index, _: index,
        ),
        Curve(
            name='ResNet',
            metrics=['Index', 'BPT'],
            run_ids=[
                'e00e610a0f3041e9a3e770f047a2a0b4',
                'fe34ae6066524e06bd774759723a2d35',
                '943b3826e79e41fd98ae9ef6dba9cd7c',
            ],
            calculate_loss=lambda *_: 0.0,
            calculate_distortion=lambda _, bpt: bpt,
            calculate_bpp=lambda index, _: resnet_index_to_layer(index),
        ),
    ),
)


RADEMACHER_BPT = Plot(
    distortion_label='Normalized Rademacher Complexity',
    rate_label='Normalized Bitrate',
    legend_label='Method: Pearson Corr.',
    filename='bpt-rademacher',
    pearson=True,
    normalize_distortion=True,
    normalize_rate=True,
    experiments=(
        Curve(
            name='GPT-2',
            metrics=['Rademacher Complexity', 'BPT'],
            run_ids=[
                'c4882b856b07487eb53238c776ad0979',
                '778eec98aa394a3888db67a4596dff8d',
                'd38c5fb7c0f54dcf90eb6e13ed48ffbb',
            ],
            calculate_loss=None,
            calculate_distortion=lambda complexity, _: complexity,
            calculate_bpp=lambda _, bpt: bpt,
        ),
        Curve(
            name='Pythia',
            metrics=['Rademacher Complexity', 'BPT'],
            run_ids=[
                '649c095e6a264baf9d3d60ea21b5dcea',
                '4c737b864bce4c18b16e7e0c16afa431',
                'b8b0148b12fc45a78a412912ea044cd4',
            ],
            calculate_loss=None,
            calculate_distortion=lambda complexity, _: complexity,
            calculate_bpp=lambda _, bpt: bpt,
        ),
        Curve(
            name='ViT',
            metrics=['Rademacher Complexity', 'BPT'],
            run_ids=[
                'bc485a43b8394bd8a6fd187388f54701',
                '05ae4381a17f46a78250d546df5ab9e7',
                'dfa9ad066ded45a887551f302a473e18',
            ],
            calculate_loss=None,
            calculate_distortion=lambda complexity, _: complexity,
            calculate_bpp=lambda _, bpt: bpt,
        ),
        Curve(
            name='ResNet',
            metrics=['Rademacher Complexity', 'BPT'],
            run_ids=[
                'e00e610a0f3041e9a3e770f047a2a0b4',
                'fe34ae6066524e06bd774759723a2d35',
                '943b3826e79e41fd98ae9ef6dba9cd7c',
            ],
            calculate_loss=None,
            calculate_distortion=lambda complexity, _: complexity,
            calculate_bpp=lambda _, bpt: bpt,
        ),
    ),
)

LIPSCHITZ_LAYER = Plot(
    distortion_label='Log Spectral Norm',
    rate_label='Split Point',
    filename='layer-lipschitz',
    experiments=(
        Curve(
            name='GPT-2',
            metrics=['Spectral Norm', 'Index'],
            run_ids=[
                '5c5fcf4510e94726b9408b2b1ed49d8a',
                '524739f57e2a4ae490534c75e4ee52e1',
                '4bf42e1f58664963aaed9702997969b6',
            ],
            calculate_loss=None,
            calculate_distortion=lambda spectral_norm, _: spectral_norm,
            calculate_bpp=lambda _, index: index,
        ),
        Curve(
            name='Pythia',
            metrics=['Spectral Norm', 'Index'],
            run_ids=[
                '2d86f244f4944da79f900403c67a80fd',
                '96fb6c12e3f6451f8eea5092518a511a',
                '68ee1832b5c4418f9b0482baf34a7303',
            ],
            calculate_loss=None,
            calculate_distortion=lambda spectral_norm, _: spectral_norm,
            calculate_bpp=lambda _, index: index,
        ),
        Curve(
            name='ViT',
            metrics=['Spectral Norm', 'Index'],
            run_ids=[
                '490142e7477640b8b31a034f0c80bf2f',
                '27ec31cb79384d898f54a8af6fe194d1',
                '9a702469041f4cd9b962619bb249d028',
            ],
            calculate_loss=None,
            calculate_distortion=lambda spectral_norm, _: spectral_norm,
            calculate_bpp=lambda _, index: index,
        ),
        Curve(
            name='ResNet',
            metrics=['Spectral Norm', 'Index'],
            run_ids=[
                'a65bec7b4008429a92b8741a86bb6858',
                '9220c408739342c9a8c1dfbac232888a',
                '13aef69aaadb441f9617eb9e855ae85a',
            ],
            calculate_loss=None,
            calculate_distortion=lambda spectral_norm, _: spectral_norm,
            calculate_bpp=lambda _, index: resnet_index_to_layer(index),
        ),
    ),
)


LIPSCHITZ_BPT = Plot(
    distortion_label='Log Spectral Norm',
    rate_label='Normalized Bitrate',
    legend_label='Method: Pearson Corr.',
    filename='bpt-lipschitz',
    pearson=True,
    normalize_rate=True,
    experiments=(
        Curve(
            name='GPT-2',
            metrics=['Spectral Norm', 'BPT'],
            run_ids=[
                '5c5fcf4510e94726b9408b2b1ed49d8a',
                '524739f57e2a4ae490534c75e4ee52e1',
                '4bf42e1f58664963aaed9702997969b6',
            ],
            calculate_loss=None,
            calculate_distortion=lambda spectral_norm, _: spectral_norm,
            calculate_bpp=lambda _, bpt: bpt,
        ),
        Curve(
            name='Pythia',
            metrics=['Spectral Norm', 'BPT'],
            run_ids=[
                '2d86f244f4944da79f900403c67a80fd',
                '96fb6c12e3f6451f8eea5092518a511a',
                '68ee1832b5c4418f9b0482baf34a7303',
            ],
            calculate_loss=None,
            calculate_distortion=lambda spectral_norm, _: spectral_norm,
            calculate_bpp=lambda _, bpt: bpt,
        ),
        Curve(
            name='ViT',
            metrics=['Spectral Norm', 'BPT'],
            run_ids=[
                '490142e7477640b8b31a034f0c80bf2f',
                '27ec31cb79384d898f54a8af6fe194d1',
                '9a702469041f4cd9b962619bb249d028',
            ],
            calculate_loss=None,
            calculate_distortion=lambda spectral_norm, _: spectral_norm,
            calculate_bpp=lambda _, bpt: bpt,
        ),
        Curve(
            name='ResNet',
            metrics=['Spectral Norm', 'BPT'],
            run_ids=[
                'a65bec7b4008429a92b8741a86bb6858',
                '9220c408739342c9a8c1dfbac232888a',
                '13aef69aaadb441f9617eb9e855ae85a',
            ],
            calculate_loss=None,
            calculate_distortion=lambda spectral_norm, _: spectral_norm,
            calculate_bpp=lambda _, bpt: bpt,
        ),
    ),
)


def resnet_index_to_layer(index: int) -> int:
    match index:
        case 1:
            return 3
        case 2:
            return 7
        case 3:
            return 13

    raise ValueError('Index not valid.')

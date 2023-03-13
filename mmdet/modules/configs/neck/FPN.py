neck=dict(
    type='FPN',
    in_channels=[96, 192, 384, 768],  #[256, 512, 1024, 2048],
    out_channels=256,
    num_outs=5)
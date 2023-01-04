_HELPTEXT: str = (
    "DeepFaceLab SAHED")

_DEFAULTS = dict(
    output_size=dict(
        default=256,
        info="resolution",
        datatype=int,
        rounding=16,
        min_max=(64, 640),
        group="model",
        fixed=True),

    archi_type=dict(
        default="liae",
        info="DF or LIAE",
        datatype=str,
        choices=["df", "liae"],
        gui_radio=True,
        group="model",
        fixed=True),

    u=dict(
        default=False,
        info="pixel norm",
        datatype=bool,
        group="model",
        fixed=True),
    d=dict(
        default=False,
        info="double resolution with upscale at end",
        datatype=bool,
        group="model",
        fixed=True),
    t=dict(
        default=False,
        info="src likeliness",
        datatype=bool,
        group="model",
        fixed=True),
    
    ae_dims=dict(
        default=256,
        info="bottleneck dims",
        datatype=int,
        rounding=2,
        min_max=(32, 1024),
        group="dimensions",
        fixed=True),
    e_dims=dict(
        default=64,
        info="encoder dims",
        datatype=int,
        rounding=2,
        min_max=(16, 256),
        group="dimensions",
        fixed=True),
    d_dims=dict(
        default=64,
        info="decoder dims",
        datatype=int,
        rounding=2,
        min_max=(16, 256),
        group="dimensions",
        fixed=True),
    
    # Weight management
    freeze_layers=dict(
        default="encoder decoder_both",
        info="Which layers to freeze",
        datatype=list,
        choices=["encoder", "inter", "inter_b", "inter_ab", "decoder_a", "decoder_b",
                 "decoder_both"],
        group="weights",
        fixed=False),
    load_layers=dict(
        default="encoder decoder_both",
        info="Which layers to load weights from pretrained model",
        datatype=list,
        choices=["encoder", "inter", "inter_b", "inter_ab", "decoder_a", "decoder_b",
                 "decoder_both"],
        group="weights",
        fixed=True),
)
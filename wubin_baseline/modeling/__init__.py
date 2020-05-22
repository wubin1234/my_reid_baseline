from .baseline import Baseline


def build_model(cfg, num_classes):
    model = Baseline(num_classes, cfg.MODEL.LAST_STRIDE, cfg.MODEL.PRETRAIN_PATH, cfg.MODEL.NECK, cfg.TEST.NECK_FEAT,
                     cfg.MODEL.NAME, cfg.MODEL.PRETRAIN_CHOICE)
    return model